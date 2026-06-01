// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Directory-based Lance Namespace implementation.
//!
//! This module provides a directory-based implementation of the Lance namespace
//! that stores tables as Lance datasets in a filesystem directory structure.

pub mod manifest;

use arrow::array::Float32Array;
use arrow::record_batch::RecordBatchIterator;
use arrow_ipc::reader::StreamReader;
use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::scanner::Scanner;
use lance::dataset::statistics::DatasetStatisticsExt;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{
    Dataset, MergeInsertBuilder, WhenMatched, WhenNotMatched, WhenNotMatchedBySource, WriteMode,
    WriteParams,
};
use lance::index::{DatasetIndexExt, IndexParams, vector::VectorIndexParams};
use lance::session::Session;
use lance_index::scalar::{
    BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
};
use lance_index::vector::{
    bq::RQBuildParams, hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, pq::PQBuildParams,
    sq::builder::SQBuildParams,
};
use lance_index::{IndexType, is_system_index};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_linalg::distance::MetricType;
use lance_table::io::commit::{ManifestNamingScheme, VERSIONS_DIR};
use object_store::ObjectStoreExt;
use object_store::path::Path;
use object_store::{Error as ObjectStoreError, ObjectStore as OSObjectStore, PutMode, PutOptions};
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use crate::context::DynamicContextProvider;
use lance_namespace::models::{
    AnalyzeTableQueryPlanRequest, BatchDeleteTableVersionsRequest,
    BatchDeleteTableVersionsResponse, CountTableRowsRequest, CreateNamespaceRequest,
    CreateNamespaceResponse, CreateTableIndexRequest, CreateTableIndexResponse, CreateTableRequest,
    CreateTableResponse, CreateTableScalarIndexResponse, CreateTableTagRequest,
    CreateTableTagResponse, CreateTableVersionRequest, CreateTableVersionResponse,
    DeclareTableRequest, DeclareTableResponse, DeleteTableTagRequest, DeleteTableTagResponse,
    DescribeNamespaceRequest, DescribeNamespaceResponse, DescribeTableIndexStatsRequest,
    DescribeTableIndexStatsResponse, DescribeTableRequest, DescribeTableResponse,
    DescribeTableVersionRequest, DescribeTableVersionResponse, DescribeTransactionRequest,
    DescribeTransactionResponse, DropNamespaceRequest, DropNamespaceResponse,
    DropTableIndexRequest, DropTableIndexResponse, DropTableRequest, DropTableResponse,
    ExplainTableQueryPlanRequest, FragmentStats, FragmentSummary, GetTableStatsRequest,
    GetTableStatsResponse, GetTableTagVersionRequest, GetTableTagVersionResponse, Identity,
    IndexContent, InsertIntoTableRequest, InsertIntoTableResponse, ListNamespacesRequest,
    ListNamespacesResponse, ListTableIndicesRequest, ListTableIndicesResponse,
    ListTableTagsRequest, ListTableTagsResponse, ListTableVersionsRequest,
    ListTableVersionsResponse, ListTablesRequest, ListTablesResponse, MergeInsertIntoTableRequest,
    MergeInsertIntoTableResponse, NamespaceExistsRequest, QueryTableRequest,
    QueryTableRequestColumns, QueryTableRequestVector, RestoreTableRequest, RestoreTableResponse,
    TableExistsRequest, TableVersion, TagContents as ModelTagContents,
    UpdateTableSchemaMetadataRequest, UpdateTableSchemaMetadataResponse, UpdateTableTagRequest,
    UpdateTableTagResponse,
};

use lance_core::{Error, Result};
use lance_namespace::LanceNamespace;
use lance_namespace::error::NamespaceError;
use lance_namespace::schema::arrow_schema_to_json;

use crate::credentials::{
    CredentialVendor, create_credential_vendor_for_location, has_credential_vendor_config,
};

/// Thread-safe metrics tracker for namespace operations.
///
/// Tracks the count of each API operation when `ops_metrics_enabled` is true.
/// Use `retrieve()` to get a snapshot of all operation counts.
#[derive(Debug, Default)]
pub struct OpsMetrics {
    counters: Mutex<HashMap<String, u64>>,
}

impl OpsMetrics {
    /// Increment the counter for an operation.
    pub fn increment(&self, operation: &str) {
        if let Ok(mut counters) = self.counters.lock() {
            *counters.entry(operation.to_string()).or_insert(0) += 1;
        }
    }

    /// Get a snapshot of all operation counts.
    pub fn retrieve(&self) -> HashMap<String, u64> {
        self.counters.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        if let Ok(mut counters) = self.counters.lock() {
            counters.clear();
        }
    }
}

/// Result of checking table status atomically.
///
/// This struct captures the state of a table directory in a single snapshot,
/// avoiding race conditions between checking existence and other status flags.
pub(crate) struct TableStatus {
    /// Whether the table directory exists (has any files)
    pub(crate) exists: bool,
    /// Whether the table has a `.lance-deregistered` marker file
    pub(crate) is_deregistered: bool,
    /// Whether the table has a `.lance-reserved` marker file (declared but not written)
    pub(crate) has_reserved_file: bool,
}

enum DirectoryIndexParams {
    Scalar {
        index_type: IndexType,
        params: ScalarIndexParams,
    },
    Inverted(InvertedIndexParams),
    Vector {
        index_type: IndexType,
        params: VectorIndexParams,
    },
}

impl DirectoryIndexParams {
    fn index_type(&self) -> IndexType {
        match self {
            Self::Scalar { index_type, .. } | Self::Vector { index_type, .. } => *index_type,
            Self::Inverted(_) => IndexType::Inverted,
        }
    }

    fn params(&self) -> &dyn IndexParams {
        match self {
            Self::Scalar { params, .. } => params,
            Self::Inverted(params) => params,
            Self::Vector { params, .. } => params,
        }
    }
}

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
    storage_options: Option<HashMap<String, String>>,
    session: Option<Arc<Session>>,
    manifest_enabled: bool,
    dir_listing_enabled: bool,
    inline_optimization_enabled: bool,
    table_version_tracking_enabled: bool,
    /// When true, table versions are stored in the `__manifest` table instead of
    /// relying on Lance's native version management.
    table_version_storage_enabled: bool,
    /// Number of manifest shards. Zero means the legacy single `__manifest` table.
    manifest_shard_count: usize,
    /// Catalog branch to open when sharded manifest mode is enabled.
    catalog_branch: String,
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
                "table_version_storage_enabled",
                &self.table_version_storage_enabled,
            )
            .field("manifest_shard_count", &self.manifest_shard_count)
            .field("catalog_branch", &self.catalog_branch)
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
            inline_optimization_enabled: true,
            table_version_tracking_enabled: false, // Default to disabled
            table_version_storage_enabled: false,  // Default to disabled
            manifest_shard_count: 0,
            catalog_branch: "main".to_string(),
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

    /// Deprecated compatibility option for inline optimization of the __manifest table.
    ///
    /// Copy-on-write manifest rewrites always replace data files and maintain
    /// indexes inline. This setting is retained for callers that still pass it.
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

    /// Enable or disable table version management through the `__manifest` table.
    ///
    /// When enabled, table versions are tracked as `table_version` entries in the
    /// `__manifest` Lance table. This enables:
    /// - Centralized version tracking instead of per-table `_versions/` directories
    ///
    /// Requires `manifest_enabled` to be true.
    /// When disabled (default), version storage uses per-table storage operations.
    pub fn table_version_storage_enabled(mut self, enabled: bool) -> Self {
        self.table_version_storage_enabled = enabled;
        self
    }

    /// Configure the number of manifest shards.
    ///
    /// A value of 0 keeps the legacy single `__manifest` table. Values greater
    /// than 0 enable sharded manifest mode.
    pub fn manifest_shard_count(mut self, shard_count: usize) -> Self {
        self.manifest_shard_count = shard_count;
        self
    }

    /// Open a catalog branch in sharded manifest mode.
    pub fn catalog_branch(mut self, branch: impl Into<String>) -> Self {
        self.catalog_branch = branch.into();
        self
    }

    /// Create a DirectoryNamespaceBuilder from properties HashMap.
    ///
    /// This method parses a properties map into builder configuration.
    /// It expects:
    /// - `root`: The root directory path (required)
    /// - `manifest_enabled`: Enable manifest-based table tracking (optional, default: true)
    /// - `dir_listing_enabled`: Enable directory listing for table discovery (optional, default: true)
    /// - `inline_optimization_enabled`: Deprecated compatibility option; ignored by copy-on-write manifest rewrites
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
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true);

        // Extract dir_listing_enabled (default: true)
        let dir_listing_enabled = properties
            .get("dir_listing_enabled")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true);

        // Extract deprecated inline_optimization_enabled (default: true)
        let inline_optimization_enabled = properties
            .get("inline_optimization_enabled")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true);

        // Extract table_version_tracking_enabled (default: false)
        let table_version_tracking_enabled = properties
            .get("table_version_tracking_enabled")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false);

        // Extract table_version_storage_enabled (default: false)
        let table_version_storage_enabled = properties
            .get("table_version_storage_enabled")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false);

        let manifest_shard_count = properties
            .get("manifest_shard_count")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);

        let catalog_branch = properties
            .get("catalog_branch")
            .cloned()
            .unwrap_or_else(|| "main".to_string());

        // Extract dir_listing_to_manifest_migration_enabled (default: false)
        let dir_listing_to_manifest_migration_enabled = properties
            .get("dir_listing_to_manifest_migration_enabled")
            .and_then(|v| v.parse::<bool>().ok())
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
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false);

        // Extract vend_input_storage_options_refresh_interval_millis (optional)
        let vend_input_storage_options_refresh_interval_millis = properties
            .get("vend_input_storage_options_refresh_interval_millis")
            .and_then(|v| v.parse::<u64>().ok());

        // Extract ops_metrics_enabled (default: false)
        let ops_metrics_enabled = properties
            .get("ops_metrics_enabled")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false);

        Ok(Self {
            root: root.trim_end_matches('/').to_string(),
            storage_options,
            session,
            manifest_enabled,
            dir_listing_enabled,
            inline_optimization_enabled,
            table_version_tracking_enabled,
            table_version_storage_enabled,
            manifest_shard_count,
            catalog_branch,
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
        // Validate: table_version_storage_enabled requires manifest_enabled
        if self.table_version_storage_enabled && !self.manifest_enabled {
            return Err(NamespaceError::InvalidInput {
                message: "table_version_storage_enabled requires manifest_enabled=true".to_string(),
            }
            .into());
        }

        let (object_store, base_path) =
            Self::initialize_object_store(&self.root, &self.storage_options, &self.session).await?;

        let manifest_ns = if self.manifest_enabled {
            let manifest_result = if self.manifest_shard_count > 0 {
                manifest::ShardedManifestNamespace::from_directory(
                    self.root.clone(),
                    self.storage_options.clone(),
                    self.session.clone(),
                    object_store.clone(),
                    base_path.clone(),
                    self.dir_listing_enabled,
                    self.inline_optimization_enabled,
                    self.commit_retries,
                    self.table_version_storage_enabled,
                    self.manifest_shard_count,
                    self.catalog_branch.clone(),
                )
                .await
                .map(|ns| ManifestCatalog::Sharded(Arc::new(ns)))
            } else {
                manifest::ManifestNamespace::from_directory(
                    self.root.clone(),
                    self.storage_options.clone(),
                    self.session.clone(),
                    object_store.clone(),
                    base_path.clone(),
                    self.dir_listing_enabled,
                    self.inline_optimization_enabled,
                    self.commit_retries,
                    self.table_version_storage_enabled,
                )
                .await
                .map(|ns| ManifestCatalog::Single(Arc::new(ns)))
            };

            match manifest_result {
                Ok(ns) => Some(Arc::new(ns)),
                Err(e) if self.manifest_shard_count > 0 => {
                    return Err(e);
                }
                Err(e) => {
                    // Failed to initialize manifest namespace, fall back to directory listing only
                    log::warn!(
                        "Failed to initialize manifest namespace, falling back to directory listing only: {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

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
            manifest_ns,
            dir_listing_enabled: self.dir_listing_enabled,
            dir_listing_to_manifest_migration_enabled: self
                .dir_listing_to_manifest_migration_enabled,
            table_version_tracking_enabled: self.table_version_tracking_enabled,
            table_version_storage_enabled: self.table_version_storage_enabled,
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

/// Directory-based implementation of Lance Namespace.
///
/// This implementation stores tables as Lance datasets in a directory structure.
/// It supports local filesystems and cloud storage backends through Lance's object store.
///
/// ## Manifest-based Listing
///
enum ManifestCatalog {
    Single(Arc<manifest::ManifestNamespace>),
    Sharded(Arc<manifest::ShardedManifestNamespace>),
}

impl std::fmt::Debug for ManifestCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(ns) => f.debug_tuple("Single").field(ns).finish(),
            Self::Sharded(ns) => f.debug_tuple("Sharded").field(ns).finish(),
        }
    }
}

impl ManifestCatalog {
    async fn list_manifest_table_locations(&self) -> Result<HashSet<String>> {
        match self {
            Self::Single(ns) => ns.list_manifest_table_locations().await,
            Self::Sharded(ns) => ns.list_manifest_table_locations().await,
        }
    }

    async fn register_table(&self, name: &str, location: String) -> Result<()> {
        match self {
            Self::Single(ns) => ns.register_table(name, location).await,
            Self::Sharded(ns) => ns.register_table(name, location).await,
        }
    }

    async fn insert_into_manifest_with_metadata(
        &self,
        entries: Vec<manifest::ManifestEntry>,
    ) -> Result<()> {
        match self {
            Self::Single(ns) => ns.insert_into_manifest_with_metadata(entries).await,
            Self::Sharded(ns) => ns.insert_into_manifest_with_metadata(entries).await,
        }
    }

    #[cfg(test)]
    async fn query_table_versions(
        &self,
        object_id: &str,
        descending: bool,
        limit: Option<i32>,
    ) -> Result<Vec<(i64, String)>> {
        match self {
            Self::Single(ns) => ns.query_table_versions(object_id, descending, limit).await,
            Self::Sharded(ns) => ns.query_table_versions(object_id, descending, limit).await,
        }
    }

    #[cfg(test)]
    async fn delete_table_versions(&self, object_id: &str, ranges: &[(i64, i64)]) -> Result<i64> {
        match self {
            Self::Single(ns) => ns.delete_table_versions(object_id, ranges).await,
            Self::Sharded(ns) => {
                ns.batch_delete_table_versions_by_ranges(&[(
                    object_id.to_string(),
                    ranges.to_vec(),
                )])
                .await
            }
        }
    }

    async fn list_table_versions(
        &self,
        table_id: &[String],
        descending: bool,
        limit: Option<i32>,
    ) -> Result<ListTableVersionsResponse> {
        match self {
            Self::Single(ns) => ns.list_table_versions(table_id, descending, limit).await,
            Self::Sharded(ns) => ns.list_table_versions(table_id, descending, limit).await,
        }
    }

    async fn describe_table_version(
        &self,
        table_id: &[String],
        version: i64,
    ) -> Result<DescribeTableVersionResponse> {
        match self {
            Self::Single(ns) => ns.describe_table_version(table_id, version).await,
            Self::Sharded(ns) => ns.describe_table_version(table_id, version).await,
        }
    }

    async fn batch_delete_table_versions_by_ranges(
        &self,
        table_ranges: &[(String, Vec<(i64, i64)>)],
    ) -> Result<i64> {
        match self {
            Self::Single(ns) => ns.batch_delete_table_versions_by_ranges(table_ranges).await,
            Self::Sharded(ns) => ns.batch_delete_table_versions_by_ranges(table_ranges).await,
        }
    }

    async fn create_catalog_branch(&self, branch: &str) -> Result<()> {
        match self {
            Self::Single(_) => Err(NamespaceError::Unsupported {
                message: "catalog branches require sharded manifest mode".to_string(),
            }
            .into()),
            Self::Sharded(ns) => ns.create_catalog_branch(branch).await,
        }
    }

    async fn promote_catalog_branch(&self, branch: &str, target_branch: &str) -> Result<()> {
        match self {
            Self::Single(_) => Err(NamespaceError::Unsupported {
                message: "catalog branch promotion requires sharded manifest mode".to_string(),
            }
            .into()),
            Self::Sharded(ns) => ns.promote_catalog_branch(branch, target_branch).await,
        }
    }
}

#[async_trait]
impl LanceNamespace for ManifestCatalog {
    fn namespace_id(&self) -> String {
        match self {
            Self::Single(ns) => ns.namespace_id(),
            Self::Sharded(ns) => ns.namespace_id(),
        }
    }

    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        match self {
            Self::Single(ns) => ns.list_namespaces(request).await,
            Self::Sharded(ns) => ns.list_namespaces(request).await,
        }
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        match self {
            Self::Single(ns) => ns.describe_namespace(request).await,
            Self::Sharded(ns) => ns.describe_namespace(request).await,
        }
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        match self {
            Self::Single(ns) => ns.create_namespace(request).await,
            Self::Sharded(ns) => ns.create_namespace(request).await,
        }
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        match self {
            Self::Single(ns) => ns.drop_namespace(request).await,
            Self::Sharded(ns) => ns.drop_namespace(request).await,
        }
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        match self {
            Self::Single(ns) => ns.namespace_exists(request).await,
            Self::Sharded(ns) => ns.namespace_exists(request).await,
        }
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        match self {
            Self::Single(ns) => ns.list_tables(request).await,
            Self::Sharded(ns) => ns.list_tables(request).await,
        }
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        match self {
            Self::Single(ns) => ns.describe_table(request).await,
            Self::Sharded(ns) => ns.describe_table(request).await,
        }
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        match self {
            Self::Single(ns) => ns.table_exists(request).await,
            Self::Sharded(ns) => ns.table_exists(request).await,
        }
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        match self {
            Self::Single(ns) => ns.drop_table(request).await,
            Self::Sharded(ns) => ns.drop_table(request).await,
        }
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        request_data: Bytes,
    ) -> Result<CreateTableResponse> {
        match self {
            Self::Single(ns) => ns.create_table(request, request_data).await,
            Self::Sharded(ns) => ns.create_table(request, request_data).await,
        }
    }

    async fn declare_table(&self, request: DeclareTableRequest) -> Result<DeclareTableResponse> {
        match self {
            Self::Single(ns) => ns.declare_table(request).await,
            Self::Sharded(ns) => ns.declare_table(request).await,
        }
    }

    async fn register_table(
        &self,
        request: lance_namespace::models::RegisterTableRequest,
    ) -> Result<lance_namespace::models::RegisterTableResponse> {
        match self {
            Self::Single(ns) => LanceNamespace::register_table(ns.as_ref(), request).await,
            Self::Sharded(ns) => LanceNamespace::register_table(ns.as_ref(), request).await,
        }
    }

    async fn deregister_table(
        &self,
        request: lance_namespace::models::DeregisterTableRequest,
    ) -> Result<lance_namespace::models::DeregisterTableResponse> {
        match self {
            Self::Single(ns) => LanceNamespace::deregister_table(ns.as_ref(), request).await,
            Self::Sharded(ns) => LanceNamespace::deregister_table(ns.as_ref(), request).await,
        }
    }
}

/// When `manifest_enabled=true`, the namespace uses a special `__manifest` Lance table to track tables
/// instead of scanning the filesystem. This provides:
/// - Better performance for listing operations
/// - Ability to track table metadata
/// - Foundation for future features like namespaces and table renaming
///
/// When `dir_listing_enabled=true`, the namespace falls back to directory scanning for tables not
/// found in the manifest, enabling gradual migration.
///
/// ## Credential Vending
///
/// When credential vendor properties are configured, `describe_table` will vend temporary
/// credentials based on the table location URI. The vendor type is auto-selected:
/// - `s3://` locations use AWS STS AssumeRole
/// - `gs://` locations use GCP OAuth2 tokens
/// - `az://` locations use Azure SAS tokens
pub struct DirectoryNamespace {
    root: String,
    storage_options: Option<HashMap<String, String>>,
    session: Option<Arc<Session>>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    manifest_ns: Option<Arc<ManifestCatalog>>,
    dir_listing_enabled: bool,
    /// When true, root-level table operations check the manifest first before
    /// falling back to directory listing. When false, root-level tables skip
    /// the manifest check and use directory listing directly.
    dir_listing_to_manifest_migration_enabled: bool,
    /// When true, `describe_table` returns `managed_versioning: true` to indicate
    /// commits should go through namespace table version APIs.
    table_version_tracking_enabled: bool,
    /// When true, table versions are stored in the `__manifest` table.
    table_version_storage_enabled: bool,
    /// Credential vendor created once during initialization.
    /// Used to vend temporary credentials for table access.
    credential_vendor: Option<Arc<dyn CredentialVendor>>,
    /// Dynamic context provider for per-request context.
    /// Stored but not directly used in operations (available for future extensions).
    #[allow(dead_code)]
    context_provider: Option<Arc<dyn DynamicContextProvider>>,
    /// When true, returns input storage options in responses when no credential vendor is configured.
    vend_input_storage_options: bool,
    /// Refresh interval in milliseconds for vended input storage options.
    /// When set, expires_at_millis is added to storage options.
    vend_input_storage_options_refresh_interval_millis: Option<u64>,
    /// Operation metrics tracker, created when ops_metrics_enabled is true.
    ops_metrics: Option<Arc<OpsMetrics>>,
}

impl std::fmt::Debug for DirectoryNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.namespace_id())
    }
}

impl std::fmt::Display for DirectoryNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.namespace_id())
    }
}

/// Describes the version ranges to delete for a single table.
/// Used by `batch_delete_table_versions` and `delete_physical_version_files`.
struct TableDeleteEntry {
    table_id: Option<Vec<String>>,
    ranges: Vec<(i64, i64)>,
}

impl DirectoryNamespace {
    /// Apply pagination to a list of table names
    ///
    /// Sorts the list alphabetically and applies pagination using page_token (start_after) and limit.
    ///
    /// # Arguments
    /// * `names` - The vector of table names to paginate
    /// * `page_token` - Skip items until finding one greater than this value (start_after semantics)
    /// * `limit` - Maximum number of items to keep
    ///
    /// # Returns
    /// The next page token (last item in this page) if more results exist beyond the limit,
    /// or `None` if this is the last page.
    fn apply_pagination(
        names: &mut Vec<String>,
        page_token: Option<String>,
        limit: Option<i32>,
    ) -> Option<String> {
        // Sort alphabetically for consistent ordering
        names.sort();

        // Apply page_token filtering (start_after semantics)
        if let Some(start_after) = page_token {
            if let Some(index) = names
                .iter()
                .position(|name| name.as_str() > start_after.as_str())
            {
                names.drain(0..index);
            } else {
                names.clear();
            }
        }

        // Apply limit and compute next page token
        if let Some(limit) = limit
            && limit >= 0
        {
            let limit = limit as usize;
            if names.len() > limit {
                let next_page_token = if limit > 0 {
                    Some(names[limit - 1].clone())
                } else {
                    None
                };
                names.truncate(limit);
                return next_page_token;
            }
        }

        None
    }

    /// List tables using directory scanning (fallback method)
    async fn list_directory_tables(&self) -> Result<Vec<String>> {
        let mut tables = Vec::new();
        let entries = self
            .object_store
            .read_dir(self.base_path.clone())
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to list directory: {:?}", e),
                })
            })?;

        for entry in entries {
            let path = entry.trim_end_matches('/');
            if !path.ends_with(".lance") {
                continue;
            }

            let table_name = &path[..path.len() - 6];

            // Use atomic check to skip deregistered tables.
            let status = self.check_table_status(table_name).await;
            if status.is_deregistered {
                continue;
            }

            tables.push(table_name.to_string());
        }

        Ok(tables)
    }

    /// Validate that the namespace ID represents the root namespace
    fn validate_root_namespace_id(id: &Option<Vec<String>>) -> Result<()> {
        if let Some(id) = id
            && !id.is_empty()
        {
            return Err(NamespaceError::Unsupported {
                message: format!(
                    "Directory namespace only supports root namespace operations, but got namespace ID: {:?}. Expected empty ID.",
                    id
                ),
            }
            .into());
        }
        Ok(())
    }

    /// Extract table name from table ID
    fn table_name_from_id(id: &Option<Vec<String>>) -> Result<String> {
        let id = id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Directory namespace table ID cannot be empty".to_string(),
            })
        })?;

        if id.len() != 1 {
            return Err(NamespaceError::Unsupported {
                message: format!(
                    "Multi-level table IDs are only supported when manifest mode is enabled, but got: {:?}",
                    id
                ),
            }
            .into());
        }

        Ok(id[0].clone())
    }

    fn format_table_id(table_id: &[String]) -> String {
        format!(
            "table id '{}'",
            manifest::ManifestNamespace::str_object_id(table_id)
        )
    }

    fn format_table_id_from_request(id: &Option<Vec<String>>) -> String {
        id.as_ref()
            .map(|table_id| Self::format_table_id(table_id))
            .unwrap_or_else(|| "table id '<unknown>'".to_string())
    }

    async fn resolve_table_location(&self, id: &Option<Vec<String>>) -> Result<String> {
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = id.clone();
        describe_req.load_detailed_metadata = Some(false);

        // Use internal impl to avoid counting this as an external API call
        let describe_resp = self.describe_table_impl(describe_req).await?;

        describe_resp.location.ok_or_else(|| {
            lance_core::Error::from(NamespaceError::TableNotFound {
                message: format!("Table location not found for: {:?}", id),
            })
        })
    }

    /// Map a Lance ref-related error returned by `Dataset::tags()` operations into
    /// the appropriate `NamespaceError` for tag APIs (create/get/update/delete).
    fn map_tag_error(err: lance_core::Error, tag: &str, table_uri: &str) -> lance_core::Error {
        match err {
            lance_core::Error::RefNotFound { .. } => NamespaceError::TableTagNotFound {
                message: format!("tag '{}' for table at '{}'", tag, table_uri),
            }
            .into(),
            lance_core::Error::RefConflict { .. } => NamespaceError::TableTagAlreadyExists {
                message: format!("tag '{}' for table at '{}'", tag, table_uri),
            }
            .into(),
            lance_core::Error::InvalidRef { message } => NamespaceError::InvalidInput {
                message: format!("invalid tag '{}': {}", tag, message),
            }
            .into(),
            lance_core::Error::VersionNotFound { message } => {
                NamespaceError::TableVersionNotFound {
                    message: format!(
                        "version referenced by tag '{}' not found for table at '{}': {}",
                        tag, table_uri, message
                    ),
                }
                .into()
            }
            other => NamespaceError::Internal {
                message: format!(
                    "tag operation failed for tag '{}' on table at '{}': {}",
                    tag, table_uri, other
                ),
            }
            .into(),
        }
    }

    async fn table_has_actual_manifests(&self, table_name: &str) -> Result<bool> {
        manifest::ManifestNamespace::path_has_actual_manifests(
            &self.object_store,
            &self.table_path(table_name),
        )
        .await
    }

    async fn filter_declared_tables(
        &self,
        tables: Vec<String>,
        include_declared: bool,
    ) -> Result<Vec<String>> {
        if include_declared {
            return Ok(tables);
        }

        let mut stream = futures::stream::iter(tables.into_iter().map(|table_name| async move {
            // `include_declared=false` is an explicit opt-in. We still pay one `_versions/` probe
            // per table here so declared-state is derived from actual manifests. This is linear in
            // the total number of listed tables, but we probe a bounded number concurrently.
            if self.table_has_actual_manifests(&table_name).await? {
                Ok::<Option<String>, Error>(Some(table_name))
            } else {
                Ok::<Option<String>, Error>(None)
            }
        }))
        .buffered(manifest::DECLARED_FILTER_CONCURRENCY);

        let mut filtered = Vec::new();
        while let Some(result) = stream.next().await {
            if let Some(table_name) = result? {
                filtered.push(table_name);
            }
        }
        Ok(filtered)
    }

    fn ipc_reader_from_request_data(
        request_data: &Bytes,
        operation: &str,
    ) -> Result<(
        Box<dyn arrow::record_batch::RecordBatchReader + Send>,
        usize,
    )> {
        if request_data.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "Request data (Arrow IPC stream) is required for {}",
                    operation
                ),
            }
            .into());
        }

        let cursor = Cursor::new(request_data.as_ref());
        let stream_reader =
            StreamReader::try_new(cursor, None).map_err(|e| NamespaceError::InvalidInput {
                message: format!("Invalid Arrow IPC stream: {}", e),
            })?;
        let arrow_schema = stream_reader.schema();

        let mut num_rows = 0usize;
        let mut batches = Vec::new();
        for batch_result in stream_reader {
            let batch = batch_result.map_err(|e| NamespaceError::Internal {
                message: format!("Failed to read batch from IPC stream: {}", e),
            })?;
            num_rows += batch.num_rows();
            batches.push(batch);
        }

        let reader: Box<dyn arrow::record_batch::RecordBatchReader + Send> = if batches.is_empty() {
            let batch = arrow::record_batch::RecordBatch::new_empty(arrow_schema.clone());
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], arrow_schema))
        } else {
            let batch_results: Vec<_> = batches.into_iter().map(Ok).collect();
            Box::new(RecordBatchIterator::new(batch_results, arrow_schema))
        };

        Ok((reader, num_rows))
    }

    async fn table_uri_has_actual_manifests(&self, table_uri: &str) -> Result<bool> {
        let table_path = self.object_store_path_from_uri(table_uri)?;
        manifest::ManifestNamespace::path_has_actual_manifests(&self.object_store, &table_path)
            .await
    }

    fn object_store_path_from_uri(&self, uri: &str) -> Result<Path> {
        let registry = self
            .session
            .as_ref()
            .map(|session| session.store_registry())
            .unwrap_or_else(|| Arc::new(ObjectStoreRegistry::default()));
        ObjectStore::extract_path_from_uri(registry, uri)
    }

    fn validate_dir_only_properties(
        properties: Option<&HashMap<String, String>>,
        operation: &str,
    ) -> Result<()> {
        // Dir-only mode has no metadata catalog, so non-empty table properties would be accepted
        // and then lost. Reject them instead. Request-level storage options are different: they
        // directly affect the current write and remain supported in dir-only mode.
        if properties.is_some_and(|properties| !properties.is_empty()) {
            return Err(NamespaceError::Unsupported {
                message: format!(
                    "{} with non-empty table properties requires manifest_enabled=true",
                    operation
                ),
            }
            .into());
        }
        Ok(())
    }

    async fn write_reader_to_table(
        &self,
        table_uri: &str,
        reader: Box<dyn arrow::record_batch::RecordBatchReader + Send>,
        mode: WriteMode,
        extra_storage_options: Option<HashMap<String, String>>,
    ) -> Result<Dataset> {
        // Insert and merge-insert request models do not carry request-level storage options,
        // so these writes intentionally use the namespace-level storage options only.
        let mut merged_storage_options = self.storage_options.clone().unwrap_or_default();
        if let Some(extra_storage_options) = extra_storage_options {
            merged_storage_options.extend(extra_storage_options);
        }
        let store_params = (!merged_storage_options.is_empty()).then(|| ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                lance_io::object_store::StorageOptionsAccessor::with_static_options(
                    merged_storage_options,
                ),
            )),
            ..Default::default()
        });

        let write_params = WriteParams {
            mode,
            store_params,
            session: self.session.clone(),
            ..Default::default()
        };

        let dataset = Dataset::write(reader, table_uri, Some(write_params))
            .await
            .map_err(|e| NamespaceError::Internal {
                message: format!("Failed to write table at '{}': {}", table_uri, e),
            })?;

        Ok(dataset)
    }

    async fn list_table_versions_from_storage(
        &self,
        table_uri: &str,
        descending: bool,
        limit: Option<i32>,
    ) -> Result<Vec<TableVersion>> {
        let table_path = self.object_store_path_from_uri(table_uri)?;
        let versions_dir = table_path.clone().join(VERSIONS_DIR);
        let manifest_metas: Vec<_> = self
            .object_store
            .read_dir_all(&versions_dir, None)
            .try_collect()
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to list manifest files for table at '{}': {}",
                        table_uri, e
                    ),
                })
            })?;

        let is_v2_naming = manifest_metas
            .first()
            .is_some_and(|meta| meta.location.filename().is_some_and(|f| f.len() == 29));

        let mut table_versions: Vec<TableVersion> = manifest_metas
            .into_iter()
            .filter_map(|meta| {
                let filename = meta.location.filename()?;
                let version_str = filename.strip_suffix(".manifest")?;
                if version_str.starts_with('d') {
                    return None;
                }
                let file_version: u64 = version_str.parse().ok()?;

                let actual_version = if file_version > u64::MAX / 2 {
                    u64::MAX - file_version
                } else {
                    file_version
                };

                Some(TableVersion {
                    version: actual_version as i64,
                    manifest_path: meta.location.to_string(),
                    manifest_size: Some(meta.size as i64),
                    e_tag: meta.e_tag,
                    timestamp_millis: Some(meta.last_modified.timestamp_millis()),
                    metadata: None,
                })
            })
            .collect();

        let list_is_ordered = self.object_store.list_is_lexically_ordered;

        let needs_sort = if list_is_ordered {
            if is_v2_naming {
                !descending
            } else {
                descending
            }
        } else {
            true
        };

        if needs_sort {
            if descending {
                table_versions.sort_by(|a, b| b.version.cmp(&a.version));
            } else {
                table_versions.sort_by(|a, b| a.version.cmp(&b.version));
            }
        }

        if let Some(limit) = limit {
            table_versions.truncate(limit as usize);
        }

        Ok(table_versions)
    }

    /// Internal describe_table implementation that doesn't record metrics.
    /// Used by both the public describe_table (which records metrics) and
    /// internal callers like resolve_table_location (which shouldn't).
    async fn describe_table_impl(
        &self,
        request: DescribeTableRequest,
    ) -> Result<DescribeTableResponse> {
        let is_root_level = request.id.as_ref().is_some_and(|id| id.len() == 1);
        let skip_manifest_for_root = self.dir_listing_enabled
            && is_root_level
            && !self.dir_listing_to_manifest_migration_enabled;
        if let Some(ref manifest_ns) = self.manifest_ns
            && !skip_manifest_for_root
        {
            match manifest_ns.describe_table(request.clone()).await {
                Ok(mut response) => {
                    if let Some(ref table_uri) = response.table_uri {
                        // For backwards compatibility, only skip vending credentials when explicitly set to false
                        let vend = request.vend_credentials.unwrap_or(true);
                        let identity = request.identity.as_deref();
                        response.storage_options = self
                            .get_storage_options_for_table(table_uri, vend, identity)
                            .await?;
                    }
                    // Set managed_versioning flag when table_version_tracking_enabled
                    if self.table_version_tracking_enabled {
                        response.managed_versioning = Some(true);
                    }
                    return Ok(response);
                }
                Err(_) if self.dir_listing_enabled && is_root_level => {
                    // Fall through to directory check only for single-level IDs
                }
                Err(e) => return Err(e),
            }
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_id = Self::format_table_id_from_request(&request.id);
        let table_uri = self.table_full_uri(&table_name);

        // Atomically check table existence and deregistration status
        let status = self.check_table_status(&table_name).await;

        if !status.exists {
            return Err(NamespaceError::TableNotFound {
                message: table_id.clone(),
            }
            .into());
        }

        if status.is_deregistered {
            return Err(NamespaceError::TableNotFound {
                message: format!("Table is deregistered: {}", table_id),
            }
            .into());
        }

        let load_detailed_metadata = request.load_detailed_metadata.unwrap_or(false);
        let should_check_declared =
            load_detailed_metadata || request.check_declared.unwrap_or(false);
        // For backwards compatibility, only skip vending credentials when explicitly set to false
        let vend_credentials = request.vend_credentials.unwrap_or(true);
        let identity = request.identity.as_deref();
        let is_only_declared = if should_check_declared {
            if status.has_reserved_file {
                Some(!self.table_has_actual_manifests(&table_name).await?)
            } else {
                Some(false)
            }
        } else {
            None
        };

        if !load_detailed_metadata {
            let storage_options = self
                .get_storage_options_for_table(&table_uri, vend_credentials, identity)
                .await?;
            return Ok(DescribeTableResponse {
                table: Some(table_name),
                namespace: request.id.as_ref().map(|id| {
                    if id.len() > 1 {
                        id[..id.len() - 1].to_vec()
                    } else {
                        vec![]
                    }
                }),
                location: Some(table_uri.clone()),
                table_uri: Some(table_uri),
                storage_options,
                is_only_declared,
                managed_versioning: if self.table_version_tracking_enabled {
                    Some(true)
                } else {
                    None
                },
                ..Default::default()
            });
        }

        if is_only_declared == Some(true) {
            let storage_options = self
                .get_storage_options_for_table(&table_uri, vend_credentials, identity)
                .await?;
            return Ok(DescribeTableResponse {
                table: Some(table_name),
                namespace: request.id.as_ref().map(|id| {
                    if id.len() > 1 {
                        id[..id.len() - 1].to_vec()
                    } else {
                        vec![]
                    }
                }),
                location: Some(table_uri.clone()),
                table_uri: Some(table_uri),
                storage_options,
                is_only_declared,
                managed_versioning: if self.table_version_tracking_enabled {
                    Some(true)
                } else {
                    None
                },
                ..Default::default()
            });
        }

        // Try to load the dataset to get real information
        // Use DatasetBuilder with storage options to support S3 with custom endpoints
        let mut builder = DatasetBuilder::from_uri(&table_uri);
        if let Some(opts) = &self.storage_options {
            builder = builder.with_storage_options(opts.clone());
        }
        if let Some(sess) = &self.session {
            builder = builder.with_session(sess.clone());
        }
        match builder.load().await {
            Ok(mut dataset) => {
                // If a specific version is requested, checkout that version
                if let Some(requested_version) = request.version {
                    dataset = dataset
                        .checkout_version(requested_version as u64)
                        .await
                        .map_err(|e| {
                            lance_core::Error::from(NamespaceError::TableVersionNotFound {
                                message: format!(
                                    "Version {} not found for table '{}': {}",
                                    requested_version, table_name, e
                                ),
                            })
                        })?;
                }

                let version_info = dataset.version();
                let lance_schema = dataset.schema();
                let arrow_schema: arrow_schema::Schema = lance_schema.into();
                let json_schema = arrow_schema_to_json(&arrow_schema)?;
                let storage_options = self
                    .get_storage_options_for_table(&table_uri, vend_credentials, identity)
                    .await?;

                // Convert BTreeMap to HashMap for the response
                let metadata: std::collections::HashMap<String, String> =
                    version_info.metadata.into_iter().collect();

                Ok(DescribeTableResponse {
                    table: Some(table_name),
                    namespace: request.id.as_ref().map(|id| {
                        if id.len() > 1 {
                            id[..id.len() - 1].to_vec()
                        } else {
                            vec![]
                        }
                    }),
                    version: Some(version_info.version as i64),
                    location: Some(table_uri.clone()),
                    table_uri: Some(table_uri),
                    schema: Some(Box::new(json_schema)),
                    storage_options,
                    metadata: Some(metadata),
                    is_only_declared,
                    managed_versioning: if self.table_version_tracking_enabled {
                        Some(true)
                    } else {
                        None
                    },
                    ..Default::default()
                })
            }
            Err(err) => {
                if manifest::ManifestNamespace::is_not_found_load_error(&err)
                    && is_only_declared == Some(true)
                {
                    let storage_options = self
                        .get_storage_options_for_table(&table_uri, vend_credentials, identity)
                        .await?;
                    Ok(DescribeTableResponse {
                        table: Some(table_name),
                        namespace: request.id.as_ref().map(|id| {
                            if id.len() > 1 {
                                id[..id.len() - 1].to_vec()
                            } else {
                                vec![]
                            }
                        }),
                        location: Some(table_uri.clone()),
                        table_uri: Some(table_uri),
                        storage_options,
                        is_only_declared,
                        managed_versioning: if self.table_version_tracking_enabled {
                            Some(true)
                        } else {
                            None
                        },
                        ..Default::default()
                    })
                } else {
                    Err(NamespaceError::Internal {
                        message: format!(
                            "Table directory exists but cannot load dataset {}: {:?}",
                            table_name, err
                        ),
                    }
                    .into())
                }
            }
        }
    }

    async fn load_dataset(
        &self,
        table_uri: &str,
        version: Option<i64>,
        operation: &str,
    ) -> Result<Dataset> {
        if let Some(version) = version
            && version < 0
        {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "Table version for {} must be non-negative, got {}",
                    operation, version
                ),
            }
            .into());
        }

        let mut builder = DatasetBuilder::from_uri(table_uri);
        if let Some(opts) = &self.storage_options {
            builder = builder.with_storage_options(opts.clone());
        }
        if let Some(sess) = &self.session {
            builder = builder.with_session(sess.clone());
        }

        let dataset = builder.load().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::TableNotFound {
                message: format!(
                    "Failed to open table at '{}' for {}: {}",
                    table_uri, operation, e
                ),
            })
        })?;

        if let Some(version) = version {
            return dataset.checkout_version(version as u64).await.map_err(|e| {
                lance_core::Error::from(NamespaceError::TableVersionNotFound {
                    message: format!(
                        "Failed to checkout version {} for table at '{}' during {}: {}",
                        version, table_uri, operation, e
                    ),
                })
            });
        }

        Ok(dataset)
    }

    fn parse_index_type(index_type: &str) -> Result<IndexType> {
        match index_type.trim().to_ascii_uppercase().as_str() {
            "SCALAR" | "BTREE" => Ok(IndexType::BTree),
            "BITMAP" => Ok(IndexType::Bitmap),
            "LABEL_LIST" | "LABELLIST" => Ok(IndexType::LabelList),
            "INVERTED" | "FTS" => Ok(IndexType::Inverted),
            "NGRAM" => Ok(IndexType::NGram),
            "ZONEMAP" | "ZONE_MAP" => Ok(IndexType::ZoneMap),
            "BLOOMFILTER" | "BLOOM_FILTER" => Ok(IndexType::BloomFilter),
            "RTREE" | "R_TREE" => Ok(IndexType::RTree),
            "VECTOR" | "IVF_PQ" => Ok(IndexType::IvfPq),
            "IVF_FLAT" => Ok(IndexType::IvfFlat),
            "IVF_SQ" => Ok(IndexType::IvfSq),
            "IVF_RQ" => Ok(IndexType::IvfRq),
            "IVF_HNSW_FLAT" => Ok(IndexType::IvfHnswFlat),
            "IVF_HNSW_SQ" => Ok(IndexType::IvfHnswSq),
            "IVF_HNSW_PQ" => Ok(IndexType::IvfHnswPq),
            other => Err(NamespaceError::InvalidInput {
                message: format!("Unsupported index_type '{}'", other),
            }
            .into()),
        }
    }

    fn parse_metric_type(distance_type: Option<&str>) -> Result<MetricType> {
        let distance_type = distance_type.unwrap_or("l2");
        MetricType::try_from(distance_type).map_err(|e| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: format!(
                    "Unsupported distance_type '{}' for vector index: {}",
                    distance_type, e
                ),
            })
        })
    }

    fn build_index_params(request: &CreateTableIndexRequest) -> Result<DirectoryIndexParams> {
        let index_type = Self::parse_index_type(&request.index_type)?;
        Ok(match index_type {
            IndexType::BTree => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            },
            IndexType::Bitmap => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
            },
            IndexType::LabelList => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList),
            },
            IndexType::NGram => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::NGram),
            },
            IndexType::ZoneMap => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
            },
            IndexType::BloomFilter => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::BloomFilter),
            },
            IndexType::RTree => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::RTree),
            },
            IndexType::Inverted => {
                let mut params = InvertedIndexParams::default();
                if let Some(with_position) = request.with_position {
                    params = params.with_position(with_position);
                }
                if let Some(base_tokenizer) = &request.base_tokenizer {
                    params = params.base_tokenizer(base_tokenizer.clone());
                }
                if let Some(language) = &request.language {
                    params = params.language(language)?;
                }
                if let Some(max_token_length) = request.max_token_length {
                    if max_token_length < 0 {
                        return Err(NamespaceError::InvalidInput {
                            message: format!(
                                "FTS max_token_length must be non-negative, got {}",
                                max_token_length
                            ),
                        }
                        .into());
                    }
                    params = params.max_token_length(Some(max_token_length as usize));
                }
                if let Some(lower_case) = request.lower_case {
                    params = params.lower_case(lower_case);
                }
                if let Some(stem) = request.stem {
                    params = params.stem(stem);
                }
                if let Some(remove_stop_words) = request.remove_stop_words {
                    params = params.remove_stop_words(remove_stop_words);
                }
                if let Some(ascii_folding) = request.ascii_folding {
                    params = params.ascii_folding(ascii_folding);
                }
                DirectoryIndexParams::Inverted(params)
            }
            IndexType::IvfFlat => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_flat_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                ),
            },
            IndexType::IvfPq => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_pq_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    PQBuildParams::default(),
                ),
            },
            IndexType::IvfSq => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_sq_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    SQBuildParams::default(),
                ),
            },
            IndexType::IvfRq => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_rq_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    RQBuildParams::default(),
                ),
            },
            IndexType::IvfHnswFlat => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::ivf_hnsw(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    HnswBuildParams::default(),
                ),
            },
            IndexType::IvfHnswSq => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_hnsw_sq_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    HnswBuildParams::default(),
                    SQBuildParams::default(),
                ),
            },
            IndexType::IvfHnswPq => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_hnsw_pq_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    HnswBuildParams::default(),
                    PQBuildParams::default(),
                ),
            },
            other => {
                return Err(NamespaceError::InvalidInput {
                    message: format!("Unsupported index type for namespace API: {}", other),
                }
                .into());
            }
        })
    }

    fn paginate_indices(
        indices: &mut Vec<IndexContent>,
        page_token: Option<String>,
        limit: Option<i32>,
    ) -> Option<String> {
        indices.sort_by(|a, b| a.index_name.cmp(&b.index_name));

        if let Some(start_after) = page_token {
            if let Some(index) = indices
                .iter()
                .position(|index| index.index_name.as_str() > start_after.as_str())
            {
                indices.drain(0..index);
            } else {
                indices.clear();
            }
        }

        let mut next_page_token = None;
        if let Some(limit) = limit
            && limit >= 0
        {
            let limit = limit as usize;
            if limit > 0 && indices.len() > limit {
                next_page_token = Some(indices[limit - 1].index_name.clone());
            }
            indices.truncate(limit);
        }
        if indices.is_empty() {
            None
        } else {
            next_page_token
        }
    }

    fn transaction_operation_name(transaction: &Transaction) -> String {
        match &transaction.operation {
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } if new_indices.is_empty() && !removed_indices.is_empty() => "DropIndex".to_string(),
            _ => transaction.operation.to_string(),
        }
    }

    fn transaction_response(
        version: u64,
        transaction: &Transaction,
    ) -> DescribeTransactionResponse {
        let mut properties = transaction
            .transaction_properties
            .as_ref()
            .map(|properties| (**properties).clone())
            .unwrap_or_default();
        properties.insert("uuid".to_string(), transaction.uuid.clone());
        properties.insert("version".to_string(), version.to_string());
        properties.insert(
            "read_version".to_string(),
            transaction.read_version.to_string(),
        );
        properties.insert(
            "operation".to_string(),
            Self::transaction_operation_name(transaction),
        );
        if let Some(tag) = &transaction.tag {
            properties.insert("tag".to_string(), tag.clone());
        }

        DescribeTransactionResponse {
            status: "SUCCEEDED".to_string(),
            properties: Some(properties),
        }
    }

    fn describe_table_index_stats_response(
        stats: &serde_json::Value,
    ) -> DescribeTableIndexStatsResponse {
        let get_i64 = |key: &str| {
            stats.get(key).and_then(|value| {
                value
                    .as_i64()
                    .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
            })
        };

        DescribeTableIndexStatsResponse {
            distance_type: stats
                .get("distance_type")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            index_type: stats
                .get("index_type")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            num_indexed_rows: get_i64("num_indexed_rows"),
            num_unindexed_rows: get_i64("num_unindexed_rows"),
            num_indices: get_i64("num_indices").and_then(|value| i32::try_from(value).ok()),
        }
    }

    /// When transaction_id is not parseable as a version number (i.e. it's a UUID),
    /// find_transaction iterates through every version in reverse, reading each
    /// transaction file from storage. For tables with many versions this will
    /// be extremely slow — each iteration is a separate I/O call.
    async fn find_transaction(&self, dataset: &Dataset, id: &str) -> Result<(u64, Transaction)> {
        if let Ok(version) = id.parse::<u64>() {
            let transaction = dataset
                .read_transaction_by_version(version)
                .await
                .map_err(|e| {
                    lance_core::Error::from(NamespaceError::TransactionNotFound {
                        message: format!(
                            "Failed to read transaction for version {}: {}",
                            version, e
                        ),
                    })
                })?
                .ok_or_else(|| {
                    lance_core::Error::from(NamespaceError::TransactionNotFound {
                        message: format!("version {}", version),
                    })
                })?;
            return Ok((version, transaction));
        }

        let versions = dataset.versions().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!(
                    "Failed to list table versions while resolving transaction '{}': {}",
                    id, e
                ),
            })
        })?;

        for version in versions.into_iter().rev() {
            if let Some(transaction) = dataset
                .read_transaction_by_version(version.version)
                .await
                .map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!(
                            "Failed to read transaction for version {} while resolving '{}': {}",
                            version.version, id, e
                        ),
                    })
                })?
                && transaction.uuid == id
            {
                return Ok((version.version, transaction));
            }
        }

        Err(NamespaceError::TransactionNotFound {
            message: id.to_string(),
        }
        .into())
    }

    fn table_full_uri(&self, table_name: &str) -> String {
        format!("{}/{}.lance", &self.root, table_name)
    }

    /// Get the object store path for a table (relative to base_path)
    fn table_path(&self, table_name: &str) -> Path {
        self.base_path
            .clone()
            .join(format!("{}.lance", table_name).as_str())
    }

    /// Get the reserved file path for a table
    fn table_reserved_file_path(&self, table_name: &str) -> Path {
        self.base_path
            .clone()
            .join(format!("{}.lance", table_name).as_str())
            .join(".lance-reserved")
    }

    /// Get the deregistered marker file path for a table
    fn table_deregistered_file_path(&self, table_name: &str) -> Path {
        self.base_path
            .clone()
            .join(format!("{}.lance", table_name).as_str())
            .join(".lance-deregistered")
    }

    /// Atomically check table existence and deregistration status.
    ///
    /// This performs a single directory listing to get a consistent snapshot of the
    /// table's state, avoiding race conditions between checking existence and
    /// checking deregistration status.
    pub(crate) async fn check_table_status(&self, table_name: &str) -> TableStatus {
        let table_path = self.table_path(table_name);
        match self.object_store.read_dir(table_path).await {
            Ok(entries) => {
                let exists = !entries.is_empty();
                let is_deregistered = entries.iter().any(|e| e.ends_with(".lance-deregistered"));
                let has_reserved_file = entries.iter().any(|e| e.ends_with(".lance-reserved"));
                TableStatus {
                    exists,
                    is_deregistered,
                    has_reserved_file,
                }
            }
            Err(_) => TableStatus {
                exists: false,
                is_deregistered: false,
                has_reserved_file: false,
            },
        }
    }

    async fn put_marker_file_atomic(
        &self,
        path: &Path,
        file_description: &str,
    ) -> std::result::Result<(), String> {
        let put_opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };

        match self
            .object_store
            .inner
            .put_opts(path, bytes::Bytes::new().into(), put_opts)
            .await
        {
            Ok(_) => Ok(()),
            Err(ObjectStoreError::AlreadyExists { .. })
            | Err(ObjectStoreError::Precondition { .. }) => {
                Err(format!("{} already exists", file_description))
            }
            Err(e) => Err(format!("Failed to create {}: {:?}", file_description, e)),
        }
    }

    /// Get storage options for a table, using credential vending if configured.
    ///
    /// If credential vendor properties are configured and the table location matches
    /// a supported cloud provider, this will create an appropriate vendor and vend
    /// temporary credentials scoped to the table location. Otherwise, returns the
    /// static storage options.
    ///
    /// The vendor type is auto-selected based on the table URI:
    /// - `s3://` locations use AWS STS AssumeRole
    /// - `gs://` locations use GCP OAuth2 tokens
    /// - `az://` locations use Azure SAS tokens
    ///
    /// The permission level (Read, Write, Admin) is configured at namespace
    /// initialization time via the `credential_vendor_permission` property.
    ///
    /// # Arguments
    ///
    /// * `table_uri` - The full URI of the table
    /// * `identity` - Optional identity from the request for identity-based credential vending
    async fn get_storage_options_for_table(
        &self,
        table_uri: &str,
        vend_credentials: bool,
        identity: Option<&Identity>,
    ) -> Result<Option<HashMap<String, String>>> {
        if vend_credentials && let Some(ref vendor) = self.credential_vendor {
            let vended = vendor.vend_credentials(table_uri, identity).await?;
            return Ok(Some(vended.storage_options));
        }
        // When vend_input_storage_options is enabled and no credential vendor is configured,
        // return the input storage options. This is useful for testing.
        if self.vend_input_storage_options {
            let mut options = self.storage_options.clone().unwrap_or_default();
            // Add expires_at_millis if refresh interval is configured
            if let Some(refresh_interval_millis) =
                self.vend_input_storage_options_refresh_interval_millis
            {
                let now_millis = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                let expires_at_millis = now_millis + refresh_interval_millis;
                options.insert(
                    "expires_at_millis".to_string(),
                    expires_at_millis.to_string(),
                );
            }
            return Ok(Some(options));
        }
        // When no credential vendor is configured, return None to avoid
        // leaking the namespace's own static credentials to clients.
        Ok(None)
    }

    /// Migrate directory-based tables to the manifest.
    ///
    /// This is a one-time migration operation that:
    /// 1. Scans the directory for existing `.lance` tables
    /// 2. Registers any unmigrated tables in the manifest
    /// 3. Returns the count of tables that were migrated
    ///
    /// This method is safe to run multiple times - it will skip tables that are already
    /// registered in the manifest.
    ///
    /// # Usage
    ///
    /// After creating tables in directory-only mode or dual mode, you can migrate them
    /// to the manifest to enable manifest-only mode:
    ///
    /// ```no_run
    /// # use lance_namespace_impls::DirectoryNamespaceBuilder;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create namespace with dual mode (manifest + directory listing)
    /// let namespace = DirectoryNamespaceBuilder::new("/path/to/data")
    ///     .manifest_enabled(true)
    ///     .dir_listing_enabled(true)
    ///     .build()
    ///     .await?;
    ///
    /// // ... tables are created and used ...
    ///
    /// // Migrate existing directory tables to manifest
    /// let migrated_count = namespace.migrate().await?;
    /// println!("Migrated {} tables", migrated_count);
    ///
    /// // Now you can disable directory listing for better performance:
    /// // (requires rebuilding the namespace)
    /// let namespace = DirectoryNamespaceBuilder::new("/path/to/data")
    ///     .manifest_enabled(true)
    ///     .dir_listing_enabled(false)  // All tables now in manifest
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Returns
    ///
    /// Returns the number of tables that were migrated to the manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Manifest is not enabled
    /// - Directory listing fails
    /// - Manifest registration fails
    pub async fn migrate(&self) -> Result<usize> {
        // We only care about tables in the root namespace
        let Some(ref manifest_ns) = self.manifest_ns else {
            return Ok(0); // No manifest, nothing to migrate
        };

        // Get all table locations already in the manifest
        let manifest_locations = manifest_ns.list_manifest_table_locations().await?;

        // Get all tables from directory and skip declared-only tables that have not
        // written any actual version manifests yet.
        let dir_tables = self
            .filter_declared_tables(self.list_directory_tables().await?, false)
            .await?;

        // Register each directory table that doesn't have an overlapping location
        // If a directory name already exists in the manifest,
        // that means the table must have already been migrated or created
        // in the manifest, so we can skip it.
        let mut migrated_count = 0;
        for table_name in dir_tables {
            // For root namespace tables, the directory name is "table_name.lance"
            let dir_name = format!("{}.lance", table_name);
            if !manifest_locations.contains(&dir_name) {
                manifest_ns.register_table(&table_name, dir_name).await?;
                migrated_count += 1;
            }
        }

        Ok(migrated_count)
    }

    /// Delete physical manifest files for the given table version ranges (best-effort).
    ///
    /// This helper is used by `batch_delete_table_versions` in both the manifest-enabled
    /// and non-manifest paths. It resolves each table's storage location, computes the
    /// version file paths, and attempts to delete them. Errors are logged (best-effort)
    /// when `best_effort` is true, or returned immediately when false.
    ///
    /// Returns the number of files successfully deleted.
    async fn delete_physical_version_files(
        &self,
        table_entries: &[TableDeleteEntry],
        best_effort: bool,
    ) -> Result<i64> {
        let mut deleted_count = 0i64;
        for te in table_entries {
            let table_uri = self.resolve_table_location(&te.table_id).await?;
            let table_path = self.object_store_path_from_uri(&table_uri)?;
            let versions_dir_path = table_path.clone().join(VERSIONS_DIR);

            for (start, end) in &te.ranges {
                for version in *start..=*end {
                    let version_path = versions_dir_path
                        .clone()
                        .join(format!("{}.manifest", version as u64));
                    match self.object_store.inner.delete(&version_path).await {
                        Ok(_) => {
                            deleted_count += 1;
                        }
                        Err(object_store::Error::NotFound { .. }) => {}
                        Err(e) => {
                            if best_effort {
                                log::warn!(
                                    "Failed to delete manifest file for version {} of table {:?}: {:?}",
                                    version,
                                    te.table_id,
                                    e
                                );
                            } else {
                                return Err(NamespaceError::Internal {
                                    message: format!(
                                        "Failed to delete version {} for table at '{}': {}",
                                        version, table_uri, e
                                    ),
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }
        Ok(deleted_count)
    }

    /// Apply all query parameters from a `QueryTableRequest`-like source onto a `Scanner`.
    ///
    /// This covers vector search, filters, column projection, limits, and ANN tuning knobs so
    /// that `explain_table_query_plan` / `analyze_table_query_plan` produce an accurate plan.
    #[allow(clippy::too_many_arguments)]
    fn apply_query_params_to_scanner(
        scanner: &mut Scanner,
        filter: Option<&str>,
        columns: Option<&QueryTableRequestColumns>,
        vector_column: Option<&str>,
        vector: &QueryTableRequestVector,
        k: i32,
        offset: Option<i32>,
        prefilter: Option<bool>,
        bypass_vector_index: Option<bool>,
        nprobes: Option<i32>,
        ef: Option<i32>,
        refine_factor: Option<i32>,
        distance_type: Option<&str>,
        fast_search_flag: Option<bool>,
        with_row_id: Option<bool>,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
        operation: &str,
    ) -> Result<()> {
        // prefilter must be set before nearest() so the fragment-scan guard sees it.
        if let Some(pf) = prefilter {
            scanner.prefilter(pf);
        }

        if let Some(filter) = filter {
            scanner.filter(filter).map_err(|e| {
                Error::invalid_input_source(
                    format!("Invalid filter expression for {}: {}", operation, e).into(),
                )
            })?;
        }

        if let Some(cols) = columns {
            if let Some(ref names) = cols.column_names {
                scanner.project(names.as_slice()).map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid column projection for {}: {}", operation, e).into(),
                    )
                })?;
            } else if let Some(ref aliases) = cols.column_aliases {
                // aliases maps output_alias -> source_column
                let pairs: Vec<(&str, &str)> = aliases
                    .iter()
                    .map(|(alias, src)| (alias.as_str(), src.as_str()))
                    .collect();
                scanner.project_with_transform(&pairs).map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid column aliases for {}: {}", operation, e).into(),
                    )
                })?;
            }
        }

        // Resolve query vector: prefer single_vector, fall back to first row of multi_vector.
        let query_vec: Option<Vec<f32>> = vector
            .single_vector
            .as_ref()
            .filter(|v| !v.is_empty())
            .cloned()
            .or_else(|| {
                vector
                    .multi_vector
                    .as_ref()
                    .and_then(|mv| mv.first())
                    .filter(|v| !v.is_empty())
                    .cloned()
            });

        if let Some(q_vec) = query_vec {
            let col = vector_column.unwrap_or("vector");
            let q = Arc::new(Float32Array::from(q_vec));
            scanner
                .nearest(col, q.as_ref(), k.max(1) as usize)
                .map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid vector query for {}: {}", operation, e).into(),
                    )
                })?;

            // ANN parameters — must be applied after nearest().
            if let Some(n) = nprobes {
                scanner.nprobes(n.max(1) as usize);
            }
            if let Some(e) = ef {
                scanner.ef(e.max(1) as usize);
            }
            if let Some(rf) = refine_factor {
                scanner.refine(rf.max(0) as u32);
            }
            // bypass_vector_index and fast_search are mutually exclusive; apply in order.
            if let Some(true) = bypass_vector_index {
                scanner.use_index(false);
            }
            if let Some(true) = fast_search_flag {
                scanner.fast_search();
            }
            if lower_bound.is_some() || upper_bound.is_some() {
                scanner.distance_range(lower_bound, upper_bound);
            }
            if let Some(dt) = distance_type {
                let metric = Self::parse_metric_type(Some(dt))?;
                scanner.distance_metric(metric);
            }
            // Apply offset on top of the k nearest results.
            if let Some(off) = offset.filter(|&o| o > 0) {
                scanner.limit(None, Some(off as i64)).map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid offset for {}: {}", operation, e).into(),
                    )
                })?;
            }
        } else {
            // Scalar (non-vector) query: treat k as a row LIMIT.
            let limit = if k > 0 { Some(k as i64) } else { None };
            scanner
                .limit(limit, offset.map(|o| o as i64))
                .map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid limit/offset for {}: {}", operation, e).into(),
                    )
                })?;
        }

        if let Some(true) = with_row_id {
            scanner.with_row_id();
        }

        Ok(())
    }

    /// Retrieve a snapshot of operation metrics.
    ///
    /// Returns a HashMap where keys are operation names (e.g., "list_tables", "describe_table")
    /// and values are the number of times each operation was called.
    ///
    /// Returns an empty HashMap if `ops_metrics_enabled` was false when building the namespace.
    pub fn retrieve_ops_metrics(&self) -> HashMap<String, u64> {
        self.ops_metrics
            .as_ref()
            .map(|m| m.retrieve())
            .unwrap_or_default()
    }

    /// Reset all operation metrics counters to zero.
    ///
    /// Does nothing if `ops_metrics_enabled` was false when building the namespace.
    pub fn reset_ops_metrics(&self) {
        if let Some(ref metrics) = self.ops_metrics {
            metrics.reset();
        }
    }

    /// Create a catalog branch in sharded manifest mode.
    pub async fn create_catalog_branch(&self, branch: &str) -> Result<()> {
        let Some(ref manifest_ns) = self.manifest_ns else {
            return Err(NamespaceError::Unsupported {
                message: "catalog branches require manifest mode".to_string(),
            }
            .into());
        };
        manifest_ns.create_catalog_branch(branch).await
    }

    /// Promote a catalog branch to another branch, usually `main`.
    pub async fn promote_catalog_branch(&self, branch: &str, target_branch: &str) -> Result<()> {
        let Some(ref manifest_ns) = self.manifest_ns else {
            return Err(NamespaceError::Unsupported {
                message: "catalog branch promotion requires manifest mode".to_string(),
            }
            .into());
        };
        manifest_ns
            .promote_catalog_branch(branch, target_branch)
            .await
    }

    /// Increment the counter for an operation.
    fn record_op(&self, operation: &str) {
        if let Some(ref metrics) = self.ops_metrics {
            metrics.increment(operation);
        }
    }
}

#[async_trait]
impl LanceNamespace for DirectoryNamespace {
    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        self.record_op("list_namespaces");
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.list_namespaces(request).await;
        }

        Self::validate_root_namespace_id(&request.id)?;
        Ok(ListNamespacesResponse::new(vec![]))
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        self.record_op("describe_namespace");
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.describe_namespace(request).await;
        }

        Self::validate_root_namespace_id(&request.id)?;
        #[allow(clippy::needless_update)]
        Ok(DescribeNamespaceResponse {
            properties: Some(HashMap::new()),
            ..Default::default()
        })
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        self.record_op("create_namespace");
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.create_namespace(request).await;
        }

        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Err(NamespaceError::NamespaceAlreadyExists {
                message: "root namespace".to_string(),
            }
            .into());
        }

        Err(NamespaceError::Unsupported {
            message: "Child namespaces are only supported when manifest mode is enabled"
                .to_string(),
        }
        .into())
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        self.record_op("drop_namespace");
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.drop_namespace(request).await;
        }

        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Root namespace cannot be dropped".to_string(),
            }
            .into());
        }

        Err(NamespaceError::Unsupported {
            message: "Child namespaces are only supported when manifest mode is enabled"
                .to_string(),
        }
        .into())
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        self.record_op("namespace_exists");
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.namespace_exists(request).await;
        }

        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Ok(());
        }

        Err(NamespaceError::NamespaceNotFound {
            message: "Child namespaces are only supported when manifest mode is enabled"
                .to_string(),
        }
        .into())
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        self.record_op("list_tables");
        // Validate that namespace ID is provided
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;

        // For child namespaces, always delegate to manifest (if enabled)
        if !namespace_id.is_empty() {
            if let Some(ref manifest_ns) = self.manifest_ns {
                return manifest_ns.list_tables(request).await;
            }
            return Err(NamespaceError::Unsupported {
                message: "Child namespaces are only supported when manifest mode is enabled"
                    .to_string(),
            }
            .into());
        }

        // When only manifest is enabled (no directory listing), delegate directly to manifest
        if let Some(ref manifest_ns) = self.manifest_ns
            && !self.dir_listing_enabled
        {
            return manifest_ns.list_tables(request).await;
        }

        // When both manifest and directory listing are enabled with migration mode,
        // we need to merge and deduplicate
        let mut tables = if self.manifest_ns.is_some()
            && self.dir_listing_enabled
            && self.dir_listing_to_manifest_migration_enabled
        {
            // Get all manifest table locations (for deduplication)
            let manifest_locations = if let Some(ref manifest_ns) = self.manifest_ns {
                manifest_ns.list_manifest_table_locations().await?
            } else {
                std::collections::HashSet::new()
            };

            // Get all manifest tables (without pagination for merging)
            let mut manifest_request = request.clone();
            manifest_request.limit = None;
            manifest_request.page_token = None;
            let manifest_tables = if let Some(ref manifest_ns) = self.manifest_ns {
                let manifest_response = manifest_ns.list_tables(manifest_request).await?;
                manifest_response.tables
            } else {
                vec![]
            };

            // Start with all manifest table names
            // Add directory tables that aren't already in the manifest (by location)
            let mut all_tables: Vec<String> = manifest_tables;
            let dir_tables = self.list_directory_tables().await?;
            for table_name in dir_tables {
                // Check if this table's location is already in the manifest
                // Manifest stores full URIs, so we need to check both formats
                let full_location = format!("{}/{}.lance", self.root, table_name);
                let relative_location = format!("{}.lance", table_name);
                if !manifest_locations.contains(&full_location)
                    && !manifest_locations.contains(&relative_location)
                {
                    all_tables.push(table_name);
                }
            }

            all_tables
        } else {
            self.list_directory_tables().await?
        };

        tables = self
            .filter_declared_tables(tables, request.include_declared.unwrap_or(true))
            .await?;

        // Apply sorting and pagination
        let next_page_token =
            Self::apply_pagination(&mut tables, request.page_token, request.limit);
        let mut response = ListTablesResponse::new(tables);
        response.page_token = next_page_token;
        Ok(response)
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        self.record_op("describe_table");
        self.describe_table_impl(request).await
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        self.record_op("table_exists");
        let is_root_level = request.id.as_ref().is_some_and(|id| id.len() == 1);
        let skip_manifest_for_root = self.dir_listing_enabled
            && is_root_level
            && !self.dir_listing_to_manifest_migration_enabled;
        if let Some(ref manifest_ns) = self.manifest_ns
            && !skip_manifest_for_root
        {
            match manifest_ns.table_exists(request.clone()).await {
                Ok(()) => return Ok(()),
                Err(_) if self.dir_listing_enabled && is_root_level => {
                    // Fall through to directory check only for single-level IDs
                }
                Err(e) => return Err(e),
            }
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_id = Self::format_table_id_from_request(&request.id);

        // Atomically check table existence and deregistration status
        let status = self.check_table_status(&table_name).await;

        if !status.exists {
            return Err(NamespaceError::TableNotFound {
                message: table_id.clone(),
            }
            .into());
        }

        if status.is_deregistered {
            return Err(NamespaceError::TableNotFound {
                message: format!("Table is deregistered: {}", table_id),
            }
            .into());
        }

        Ok(())
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        self.record_op("drop_table");
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.drop_table(request).await;
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);
        let table_path = self.table_path(&table_name);

        self.object_store
            .remove_dir_all(table_path)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to drop table {}: {:?}", table_name, e),
                })
            })?;

        Ok(DropTableResponse {
            id: request.id,
            location: Some(table_uri),
            ..Default::default()
        })
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        request_data: Bytes,
    ) -> Result<CreateTableResponse> {
        self.record_op("create_table");
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.create_table(request, request_data).await;
        }

        Self::validate_dir_only_properties(request.properties.as_ref(), "create_table")?;

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);
        let status = self.check_table_status(&table_name).await;
        let (reader, _num_rows) =
            Self::ipc_reader_from_request_data(&request_data, "create_table")?;

        if status.exists && self.table_has_actual_manifests(&table_name).await? {
            return Err(NamespaceError::TableAlreadyExists {
                message: table_name,
            }
            .into());
        }

        let write_result = self
            .write_reader_to_table(
                &table_uri,
                reader,
                WriteMode::Create,
                request.storage_options.clone(),
            )
            .await;
        if let Err(err) = write_result {
            if self.table_uri_has_actual_manifests(&table_uri).await? {
                return Err(NamespaceError::TableAlreadyExists {
                    message: table_name,
                }
                .into());
            }
            return Err(err);
        }
        Ok(CreateTableResponse {
            version: Some(1),
            location: Some(table_uri),
            storage_options: self.storage_options.clone(),
            properties: request.properties,
            ..Default::default()
        })
    }

    async fn declare_table(&self, request: DeclareTableRequest) -> Result<DeclareTableResponse> {
        self.record_op("declare_table");
        if let Some(ref manifest_ns) = self.manifest_ns {
            let mut response = manifest_ns.declare_table(request.clone()).await?;
            if let Some(ref location) = response.location {
                // For backwards compatibility, only skip vending credentials when explicitly set to false
                let vend = request.vend_credentials.unwrap_or(true);
                let identity = request.identity.as_deref();
                response.storage_options = self
                    .get_storage_options_for_table(location, vend, identity)
                    .await?;
            }
            // Set managed_versioning when table_version_tracking_enabled
            if self.table_version_tracking_enabled {
                response.managed_versioning = Some(true);
            }
            return Ok(response);
        }

        Self::validate_dir_only_properties(request.properties.as_ref(), "declare_table")?;

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Validate location if provided
        if let Some(location) = &request.location {
            let location = location.trim_end_matches('/');
            if location != table_uri {
                return Err(NamespaceError::InvalidInput {
                    message: format!(
                        "Cannot declare table {} at location {}, must be at location {}",
                        table_name, location, table_uri
                    ),
                }
                .into());
            }
        }

        // Check if table already has data (created via create_table).
        // The atomic put only prevents races between concurrent declare_table calls,
        // not between declare_table and existing data.
        let status = self.check_table_status(&table_name).await;
        if status.exists && !status.has_reserved_file {
            // Table has data but no reserved file - it was created with data
            return Err(NamespaceError::TableAlreadyExists {
                message: table_name.to_string(),
            }
            .into());
        }

        // Atomically create the .lance-reserved file to mark the table as declared.
        // This uses put_if_not_exists semantics to avoid race conditions between
        // concurrent declare_table calls.
        let reserved_file_path = self.table_reserved_file_path(&table_name);

        self.put_marker_file_atomic(&reserved_file_path, &format!("table {}", table_name))
            .await
            .map_err(|e| {
                if e.contains("already exists") {
                    lance_core::Error::from(NamespaceError::TableAlreadyExists {
                        message: table_name.to_string(),
                    })
                } else {
                    lance_core::Error::from(NamespaceError::Internal { message: e })
                }
            })?;

        // For backwards compatibility, only skip vending credentials when explicitly set to false
        let vend_credentials = request.vend_credentials.unwrap_or(true);
        let identity = request.identity.as_deref();
        let storage_options = self
            .get_storage_options_for_table(&table_uri, vend_credentials, identity)
            .await?;

        Ok(DeclareTableResponse {
            location: Some(table_uri),
            storage_options,
            properties: request.properties,
            managed_versioning: if self.table_version_tracking_enabled {
                Some(true)
            } else {
                None
            },
            ..Default::default()
        })
    }

    async fn register_table(
        &self,
        request: lance_namespace::models::RegisterTableRequest,
    ) -> Result<lance_namespace::models::RegisterTableResponse> {
        self.record_op("register_table");
        // If manifest is enabled, delegate to manifest namespace
        if let Some(ref manifest_ns) = self.manifest_ns {
            return LanceNamespace::register_table(manifest_ns.as_ref(), request).await;
        }

        // Without manifest, register_table is not supported
        Err(NamespaceError::Unsupported {
            message: "register_table is only supported when manifest mode is enabled".to_string(),
        }
        .into())
    }

    async fn deregister_table(
        &self,
        request: lance_namespace::models::DeregisterTableRequest,
    ) -> Result<lance_namespace::models::DeregisterTableResponse> {
        self.record_op("deregister_table");
        // If manifest is enabled, delegate to manifest namespace
        if let Some(ref manifest_ns) = self.manifest_ns {
            return LanceNamespace::deregister_table(manifest_ns.as_ref(), request).await;
        }

        // V1 mode: create a .lance-deregistered marker file in the table directory
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Check table existence and deregistration status.
        // This provides better error messages for common cases.
        let status = self.check_table_status(&table_name).await;

        if !status.exists {
            return Err(NamespaceError::TableNotFound {
                message: table_name.to_string(),
            }
            .into());
        }

        if status.is_deregistered {
            return Err(NamespaceError::TableNotFound {
                message: format!("Table is already deregistered: {}", table_name),
            }
            .into());
        }

        // Atomically create the .lance-deregistered marker file.
        // This uses put_if_not_exists semantics to prevent race conditions
        // when multiple processes try to deregister the same table concurrently.
        // If a race occurs and another process already created the file,
        // we'll get an AlreadyExists error which we convert to a proper message.
        let deregistered_path = self.table_deregistered_file_path(&table_name);
        self.put_marker_file_atomic(
            &deregistered_path,
            &format!("deregistration marker for table {}", table_name),
        )
        .await
        .map_err(|e| {
            if e.contains("already exists") {
                lance_core::Error::from(NamespaceError::InvalidTableState {
                    message: format!("Table is already deregistered: {}", table_name),
                })
            } else {
                lance_core::Error::from(NamespaceError::Internal { message: e })
            }
        })?;

        Ok(lance_namespace::models::DeregisterTableResponse {
            id: request.id,
            location: Some(table_uri),
            ..Default::default()
        })
    }

    async fn list_table_versions(
        &self,
        request: ListTableVersionsRequest,
    ) -> Result<ListTableVersionsResponse> {
        self.record_op("list_table_versions");
        // When table_version_storage_enabled, query from __manifest
        if self.table_version_storage_enabled
            && let Some(ref manifest_ns) = self.manifest_ns
        {
            let table_id = request.id.clone().unwrap_or_default();
            let want_descending = request.descending == Some(true);
            return manifest_ns
                .list_table_versions(&table_id, want_descending, request.limit)
                .await;
        }

        // Fallback when table_version_storage is not enabled: list from _versions/ directory
        let table_uri = self.resolve_table_location(&request.id).await?;
        let want_descending = request.descending == Some(true);
        let table_versions = self
            .list_table_versions_from_storage(&table_uri, want_descending, request.limit)
            .await?;

        Ok(ListTableVersionsResponse {
            versions: table_versions,
            page_token: None,
        })
    }

    async fn create_table_version(
        &self,
        request: CreateTableVersionRequest,
    ) -> Result<CreateTableVersionResponse> {
        self.record_op("create_table_version");
        let table_uri = self.resolve_table_location(&request.id).await?;

        let staging_manifest_path = &request.manifest_path;
        let version = request.version as u64;

        let table_path = self.object_store_path_from_uri(&table_uri)?;

        // Determine naming scheme from request, default to V2
        let naming_scheme = match request.naming_scheme.as_deref() {
            Some("V1") => ManifestNamingScheme::V1,
            _ => ManifestNamingScheme::V2,
        };

        // Compute final path using the naming scheme
        let final_path = naming_scheme.manifest_path(&table_path, version);

        let staging_path = Path::parse(staging_manifest_path).map_err(|e| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: format!(
                    "Invalid staging manifest path '{}': {}",
                    staging_manifest_path, e
                ),
            })
        })?;

        let copy_result = match self
            .object_store
            .inner
            .copy_if_not_exists(&staging_path, &final_path)
            .await
        {
            Ok(()) => Ok(()),
            Err(ObjectStoreError::NotImplemented { .. })
            | Err(ObjectStoreError::NotSupported { .. }) => {
                let manifest_data = self
                    .object_store
                    .inner
                    .get(&staging_path)
                    .await
                    .map_err(|e| {
                        lance_core::Error::from(NamespaceError::Internal {
                            message: format!(
                                "Failed to read staging manifest at '{}': {}",
                                staging_manifest_path, e
                            ),
                        })
                    })?
                    .bytes()
                    .await
                    .map_err(|e| {
                        lance_core::Error::from(NamespaceError::Internal {
                            message: format!(
                                "Failed to read staging manifest bytes at '{}': {}",
                                staging_manifest_path, e
                            ),
                        })
                    })?;
                self.object_store
                    .inner
                    .put_opts(
                        &final_path,
                        manifest_data.into(),
                        PutOptions {
                            mode: PutMode::Create,
                            ..Default::default()
                        },
                    )
                    .await
                    .map(|_| ())
            }
            Err(e) => Err(e),
        };

        match copy_result {
            Ok(()) => {}
            Err(ObjectStoreError::AlreadyExists { .. })
            | Err(ObjectStoreError::Precondition { .. }) => {
                return Err(lance_core::Error::from(
                    NamespaceError::ConcurrentModification {
                        message: format!(
                            "Version {} already exists for table at '{}'",
                            version, table_uri
                        ),
                    },
                ));
            }
            Err(e) => {
                return Err(lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to create version {} for table at '{}': {}",
                        version, table_uri, e
                    ),
                }));
            }
        }

        let final_meta = self
            .object_store
            .inner
            .head(&final_path)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to stat created version {} for table at '{}': {}",
                        version, table_uri, e
                    ),
                })
            })?;
        let manifest_size = final_meta.size as i64;

        // Delete the staging manifest after successful copy
        if let Err(e) = self.object_store.inner.delete(&staging_path).await {
            log::warn!(
                "Failed to delete staging manifest at '{}': {:?}",
                staging_path,
                e
            );
        }

        // If table_version_storage_enabled is enabled, also record in __manifest (best-effort)
        if self.table_version_storage_enabled
            && let Some(ref manifest_ns) = self.manifest_ns
        {
            let table_id_str =
                manifest::ManifestNamespace::str_object_id(&request.id.clone().unwrap_or_default());
            let object_id =
                manifest::ManifestNamespace::build_version_object_id(&table_id_str, version as i64);
            let metadata_json = serde_json::json!({
                "manifest_path": final_path.to_string(),
                "manifest_size": manifest_size,
                "e_tag": final_meta.e_tag,
                "naming_scheme": request.naming_scheme.as_deref().unwrap_or("V2"),
            })
            .to_string();

            if let Err(e) = manifest_ns
                .insert_into_manifest_with_metadata(vec![manifest::ManifestEntry {
                    object_id,
                    object_type: manifest::ObjectType::TableVersion,
                    location: None,
                    metadata: Some(metadata_json),
                }])
                .await
            {
                log::warn!(
                    "Failed to record table version in __manifest (best-effort): {:?}",
                    e
                );
            }
        }

        Ok(CreateTableVersionResponse {
            transaction_id: None,
            version: Some(Box::new(TableVersion {
                version: version as i64,
                manifest_path: final_path.to_string(),
                manifest_size: Some(manifest_size),
                e_tag: final_meta.e_tag,
                timestamp_millis: None,
                metadata: None,
            })),
        })
    }

    async fn describe_table_version(
        &self,
        request: DescribeTableVersionRequest,
    ) -> Result<DescribeTableVersionResponse> {
        self.record_op("describe_table_version");
        // When table_version_storage_enabled and a specific version is requested,
        // query from __manifest to avoid opening the entire dataset
        if self.table_version_storage_enabled
            && let (Some(manifest_ns), Some(version)) = (&self.manifest_ns, request.version)
        {
            let table_id = request.id.clone().unwrap_or_default();
            return manifest_ns.describe_table_version(&table_id, version).await;
        }

        // Fallback when table_version_storage is not enabled: inspect physical manifests directly.
        let table_uri = self.resolve_table_location(&request.id).await?;
        let versions = self
            .list_table_versions_from_storage(&table_uri, true, None)
            .await?;
        let table_version = if let Some(requested_version) = request.version {
            versions
                .into_iter()
                .find(|version| version.version == requested_version)
                .ok_or_else(|| {
                    lance_core::Error::from(NamespaceError::TableVersionNotFound {
                        message: format!(
                            "version {} for table {}",
                            requested_version,
                            Self::format_table_id_from_request(&request.id)
                        ),
                    })
                })?
        } else {
            versions.into_iter().next().ok_or_else(|| {
                lance_core::Error::from(NamespaceError::TableVersionNotFound {
                    message: format!(
                        "latest version for table {}",
                        Self::format_table_id_from_request(&request.id)
                    ),
                })
            })?
        };

        Ok(DescribeTableVersionResponse {
            version: Box::new(table_version),
        })
    }

    async fn batch_delete_table_versions(
        &self,
        request: BatchDeleteTableVersionsRequest,
    ) -> Result<BatchDeleteTableVersionsResponse> {
        self.record_op("batch_delete_table_versions");
        // Single-table mode: use `id` (from path parameter) + `ranges` to delete
        // versions from one table.
        let ranges: Vec<(i64, i64)> = request
            .ranges
            .iter()
            .map(|r| {
                let start = r.start_version;
                let end = if r.end_version > 0 {
                    r.end_version
                } else {
                    start
                };
                (start, end)
            })
            .collect();
        let table_entries = vec![TableDeleteEntry {
            table_id: request.id.clone(),
            ranges,
        }];

        if self.table_version_storage_enabled
            && let Some(ref manifest_ns) = self.manifest_ns
        {
            // Phase 1 (atomic commit point): Delete version records from __manifest.
            // This is the authoritative
            // source of truth — once __manifest entries are removed, the versions
            // are logically deleted.
            let table_ranges = table_entries
                .iter()
                .map(|te| {
                    (
                        manifest::ManifestNamespace::str_object_id(
                            &te.table_id.clone().unwrap_or_default(),
                        ),
                        te.ranges.clone(),
                    )
                })
                .collect::<Vec<_>>();
            let total_deleted_count = manifest_ns
                .batch_delete_table_versions_by_ranges(&table_ranges)
                .await?;

            // Phase 2: Delete physical manifest files (best-effort).
            // Even if some file deletions fail, the versions are already removed from
            // __manifest, so they won't be visible to readers. Leftover files are
            // orphaned but harmless and can be cleaned up later.
            let _ = self
                .delete_physical_version_files(&table_entries, true)
                .await;

            return Ok(BatchDeleteTableVersionsResponse {
                deleted_count: Some(total_deleted_count),
                transaction_id: None,
            });
        }

        // Fallback when table_version_storage is not enabled: delete physical files directly (no __manifest)
        let total_deleted_count = self
            .delete_physical_version_files(&table_entries, false)
            .await?;

        Ok(BatchDeleteTableVersionsResponse {
            deleted_count: Some(total_deleted_count),
            transaction_id: None,
        })
    }

    async fn create_table_index(
        &self,
        request: CreateTableIndexRequest,
    ) -> Result<CreateTableIndexResponse> {
        self.record_op("create_table_index");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let mut dataset = self
            .load_dataset(&table_uri, None, "create_table_index")
            .await?;
        let index_request = Self::build_index_params(&request)?;

        dataset
            .create_index(
                &[request.column.as_str()],
                index_request.index_type(),
                request.name.clone(),
                index_request.params(),
                false,
            )
            .await
            .map_err(|e| {
                let err_msg = format!("{}", e);
                let ns_err = if err_msg.contains("already exists") {
                    NamespaceError::TableIndexAlreadyExists {
                        message: format!(
                            "Index '{}' already exists on table '{}': {:?}",
                            request.name.as_deref().unwrap_or("<auto-generated>"),
                            table_uri,
                            e
                        ),
                    }
                } else if err_msg.contains("not found") || err_msg.contains("does not exist") {
                    NamespaceError::TableColumnNotFound {
                        message: format!(
                            "Column '{}' not found for table '{}': {:?}",
                            request.column, table_uri, e
                        ),
                    }
                } else {
                    NamespaceError::Internal {
                        message: format!(
                            "Failed to create {} index '{}' on column '{}' for table '{}': {:?}",
                            request.index_type,
                            request.name.as_deref().unwrap_or("<auto-generated>"),
                            request.column,
                            table_uri,
                            e
                        ),
                    }
                };
                lance_core::Error::from(ns_err)
            })?;

        let transaction_id = dataset
            .read_transaction()
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to read committed transaction after creating index on '{}': {}",
                        table_uri, e
                    ),
                })
            })?
            .map(|transaction| transaction.uuid);

        Ok(CreateTableIndexResponse { transaction_id })
    }

    async fn list_table_indices(
        &self,
        request: ListTableIndicesRequest,
    ) -> Result<ListTableIndicesResponse> {
        self.record_op("list_table_indices");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, request.version, "list_table_indices")
            .await?;
        let mut indices = dataset
            .describe_indices(None)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to describe table indices for '{}': {:?}", table_uri, e),
                })
            })?
            .into_iter()
            .filter(|description| {
                description
                    .metadata()
                    .first()
                    .map(|metadata| !is_system_index(metadata))
                    .unwrap_or(false)
            })
            .map(|description| {
                let columns = description
                    .field_ids()
                    .iter()
                        .map(|field_id| {
                        dataset
                            .schema()
                            .field_path(i32::try_from(*field_id).map_err(|e| {
                                lance_core::Error::from(NamespaceError::Internal {
                                    message: format!(
                                        "Field id {} does not fit in i32 for table '{}': {}",
                                        field_id, table_uri, e
                                    ),
                                })
                            })?)
                            .map_err(|e| {
                            lance_core::Error::from(NamespaceError::Internal {
                                message: format!(
                                    "Failed to resolve field path for field_id {} in table '{}': {}",
                                    field_id, table_uri, e
                                ),
                            })
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                Ok(IndexContent {
                    index_name: description.name().to_string(),
                    index_uuid: description.metadata()[0].uuid.to_string(),
                    columns,
                    status: "SUCCEEDED".to_string(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let page_token = Self::paginate_indices(&mut indices, request.page_token, request.limit);
        Ok(ListTableIndicesResponse {
            indexes: indices,
            page_token,
        })
    }

    async fn describe_table_index_stats(
        &self,
        request: DescribeTableIndexStatsRequest,
    ) -> Result<DescribeTableIndexStatsResponse> {
        self.record_op("describe_table_index_stats");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, request.version, "describe_table_index_stats")
            .await?;
        let index_name = request.index_name.as_deref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Index name is required for describe_table_index_stats".to_string(),
            })
        })?;
        let metadatas = dataset
            .load_indices_by_name(index_name)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::TableIndexNotFound {
                    message: format!(
                        "Failed to load index '{}' metadata for table '{}': {}",
                        index_name, table_uri, e
                    ),
                })
            })?;
        if metadatas.first().is_some_and(is_system_index) {
            return Err(NamespaceError::Unsupported {
                message: format!("System index '{}' is not exposed by this API", index_name),
            }
            .into());
        }

        let stats = <Dataset as DatasetIndexExt>::index_statistics(&dataset, index_name)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::TableIndexNotFound {
                    message: format!(
                        "Failed to describe index statistics for '{}' on table '{}': {}",
                        index_name, table_uri, e
                    ),
                })
            })?;
        let stats: serde_json::Value = serde_json::from_str(&stats).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!(
                    "Failed to parse index statistics for '{}' on table '{}': {}",
                    index_name, table_uri, e
                ),
            })
        })?;

        Ok(Self::describe_table_index_stats_response(&stats))
    }

    async fn describe_transaction(
        &self,
        request: DescribeTransactionRequest,
    ) -> Result<DescribeTransactionResponse> {
        self.record_op("describe_transaction");
        let mut request_id = request.id.ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Transaction id must include table id and transaction identifier"
                    .to_string(),
            })
        })?;
        if request_id.len() < 2 {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "Transaction request id must include table id and transaction identifier, got {:?}",
                    request_id
                ),
            }
            .into());
        }

        let id = request_id.pop().expect("request_id len checked above");
        let table_id = Some(request_id);
        let table_uri = self.resolve_table_location(&table_id).await?;
        let dataset = self
            .load_dataset(&table_uri, None, "describe_transaction")
            .await?;
        let (version, transaction) = self.find_transaction(&dataset, &id).await?;

        Ok(Self::transaction_response(version, &transaction))
    }

    async fn create_table_scalar_index(
        &self,
        request: CreateTableIndexRequest,
    ) -> Result<CreateTableScalarIndexResponse> {
        self.record_op("create_table_scalar_index");
        let index_type = Self::parse_index_type(&request.index_type)?;
        if !index_type.is_scalar() {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "create_table_scalar_index only supports scalar index types, got {}",
                    request.index_type
                ),
            }
            .into());
        }

        let response = self.create_table_index(request).await?;
        Ok(CreateTableScalarIndexResponse {
            transaction_id: response.transaction_id,
        })
    }

    async fn drop_table_index(
        &self,
        request: DropTableIndexRequest,
    ) -> Result<DropTableIndexResponse> {
        self.record_op("drop_table_index");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let index_name = request.index_name.as_deref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Index name is required for drop_table_index".to_string(),
            })
        })?;
        let mut dataset = self
            .load_dataset(&table_uri, None, "drop_table_index")
            .await?;
        let metadatas = dataset
            .load_indices_by_name(index_name)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::TableIndexNotFound {
                    message: format!(
                        "Failed to load index '{}' before dropping it from table '{}': {}",
                        index_name, table_uri, e
                    ),
                })
            })?;
        if metadatas.first().is_some_and(is_system_index) {
            return Err(NamespaceError::Unsupported {
                message: format!(
                    "System index '{}' cannot be dropped via this API",
                    index_name
                ),
            }
            .into());
        }

        dataset.drop_index(index_name).await.map_err(|e| {
            lance_core::Error::from(NamespaceError::TableIndexNotFound {
                message: format!(
                    "Failed to drop index '{}' from table '{}': {}",
                    index_name, table_uri, e
                ),
            })
        })?;

        let transaction_id = dataset
            .read_transaction()
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to read committed transaction after dropping index '{}' from '{}': {}",
                        index_name, table_uri, e
                    ),
                })
            })?
            .map(|transaction| transaction.uuid);

        Ok(DropTableIndexResponse { transaction_id })
    }

    async fn list_all_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        // In dir-only mode there are no child namespaces, so all tables live in the
        // root directory. This is equivalent to listing the root namespace.
        let mut tables = self.list_directory_tables().await?;
        tables = self
            .filter_declared_tables(tables, request.include_declared.unwrap_or(true))
            .await?;
        Self::apply_pagination(&mut tables, request.page_token, request.limit);
        Ok(ListTablesResponse::new(tables))
    }

    async fn restore_table(&self, request: RestoreTableRequest) -> Result<RestoreTableResponse> {
        let version = request.version;
        if version < 0 {
            return Err(Error::invalid_input_source(
                format!(
                    "Table version for restore_table must be non-negative, got {}",
                    version
                )
                .into(),
            ));
        }

        let table_uri = self.resolve_table_location(&request.id).await?;
        let mut dataset = self.load_dataset(&table_uri, None, "restore_table").await?;

        dataset = dataset
            .checkout_version(version as u64)
            .await
            .map_err(|e| {
                Error::namespace_source(
                    format!(
                        "Failed to checkout version {} for restore at '{}': {}",
                        version, table_uri, e
                    )
                    .into(),
                )
            })?;

        dataset.restore().await.map_err(|e| {
            Error::namespace_source(
                format!(
                    "Failed to restore table at '{}' to version {}: {}",
                    table_uri, version, e
                )
                .into(),
            )
        })?;

        let transaction_id = dataset
            .read_transaction()
            .await
            .map_err(|e| {
                Error::namespace_source(
                    format!(
                        "Failed to read transaction after restoring '{}': {}",
                        table_uri, e
                    )
                    .into(),
                )
            })?
            .map(|t| t.uuid);

        Ok(RestoreTableResponse { transaction_id })
    }

    async fn update_table_schema_metadata(
        &self,
        request: UpdateTableSchemaMetadataRequest,
    ) -> Result<UpdateTableSchemaMetadataResponse> {
        let table_uri = self.resolve_table_location(&request.id).await?;
        let mut dataset = self
            .load_dataset(&table_uri, None, "update_table_schema_metadata")
            .await?;

        let new_metadata = request.metadata.unwrap_or_default();
        let updated_metadata = dataset
            .update_schema_metadata(new_metadata.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .await
            .map_err(|e| {
                Error::namespace_source(
                    format!(
                        "Failed to update schema metadata for table at '{}': {}",
                        table_uri, e
                    )
                    .into(),
                )
            })?;

        let transaction_id = dataset
            .read_transaction()
            .await
            .map_err(|e| {
                Error::namespace_source(
                    format!(
                        "Failed to read transaction after updating metadata for '{}': {}",
                        table_uri, e
                    )
                    .into(),
                )
            })?
            .map(|t| t.uuid);

        Ok(UpdateTableSchemaMetadataResponse {
            metadata: Some(updated_metadata),
            transaction_id,
        })
    }

    async fn get_table_stats(
        &self,
        request: GetTableStatsRequest,
    ) -> Result<GetTableStatsResponse> {
        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = Arc::new(
            self.load_dataset(&table_uri, None, "get_table_stats")
                .await?,
        );

        // Compute total bytes on disk using field-level statistics
        let data_stats = dataset.calculate_data_stats().await.map_err(|e| {
            Error::namespace_source(
                format!(
                    "Failed to calculate data statistics for table at '{}': {}",
                    table_uri, e
                )
                .into(),
            )
        })?;
        let total_bytes: i64 = data_stats
            .fields
            .iter()
            .map(|f| f.bytes_on_disk as i64)
            .sum();

        // Collect per-fragment row counts
        let fragment_row_futures: Vec<_> = dataset
            .get_fragments()
            .into_iter()
            .map(|f| async move { f.physical_rows().await })
            .collect();
        let fragment_row_results = futures::future::join_all(fragment_row_futures).await;
        let mut fragment_row_counts: Vec<i64> = fragment_row_results
            .into_iter()
            .filter_map(|r| r.ok())
            .map(|r| r as i64)
            .collect();

        let num_fragments = fragment_row_counts.len() as i64;
        let num_rows: i64 = fragment_row_counts.iter().sum();

        // Fragments with fewer rows than the compaction target are considered "small",
        // consistent with CompactionOptions::target_rows_per_fragment default.
        const SMALL_FRAGMENT_THRESHOLD: i64 = 1024 * 1024;
        let num_small_fragments = fragment_row_counts
            .iter()
            .filter(|&&r| r < SMALL_FRAGMENT_THRESHOLD)
            .count() as i64;

        // Compute length summary statistics
        fragment_row_counts.sort_unstable();
        let lengths = if fragment_row_counts.is_empty() {
            FragmentSummary::new(0, 0, 0, 0, 0, 0, 0)
        } else {
            let len = fragment_row_counts.len();
            let min = fragment_row_counts[0];
            let max = fragment_row_counts[len - 1];
            let mean = num_rows / num_fragments;
            let pct = |p: f64| fragment_row_counts[((len - 1) as f64 * p) as usize];
            FragmentSummary::new(min, max, mean, pct(0.25), pct(0.50), pct(0.75), pct(0.99))
        };

        // Count non-system indices
        let indices = dataset.load_indices().await.map_err(|e| {
            Error::namespace_source(
                format!("Failed to load indices for table at '{}': {}", table_uri, e).into(),
            )
        })?;
        let num_indices = indices.iter().filter(|m| !is_system_index(m)).count() as i64;

        let fragment_stats = FragmentStats::new(num_fragments, num_small_fragments, lengths);
        Ok(GetTableStatsResponse::new(
            total_bytes,
            num_rows,
            num_indices,
            fragment_stats,
        ))
    }

    async fn explain_table_query_plan(
        &self,
        request: ExplainTableQueryPlanRequest,
    ) -> Result<String> {
        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(
                &table_uri,
                request.query.version,
                "explain_table_query_plan",
            )
            .await?;
        let verbose = request.verbose.unwrap_or(false);

        let mut scanner = dataset.scan();
        Self::apply_query_params_to_scanner(
            &mut scanner,
            request.query.filter.as_deref(),
            request.query.columns.as_deref(),
            request.query.vector_column.as_deref(),
            &request.query.vector,
            request.query.k,
            request.query.offset,
            request.query.prefilter,
            request.query.bypass_vector_index,
            request.query.nprobes,
            request.query.ef,
            request.query.refine_factor,
            request.query.distance_type.as_deref(),
            request.query.fast_search,
            request.query.with_row_id,
            request.query.lower_bound,
            request.query.upper_bound,
            "explain_table_query_plan",
        )?;

        scanner.explain_plan(verbose).await.map_err(|e| {
            Error::namespace_source(
                format!(
                    "Failed to explain query plan for table at '{}': {}",
                    table_uri, e
                )
                .into(),
            )
        })
    }

    async fn analyze_table_query_plan(
        &self,
        request: AnalyzeTableQueryPlanRequest,
    ) -> Result<String> {
        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, request.version, "analyze_table_query_plan")
            .await?;

        let mut scanner = dataset.scan();
        Self::apply_query_params_to_scanner(
            &mut scanner,
            request.filter.as_deref(),
            request.columns.as_deref(),
            request.vector_column.as_deref(),
            &request.vector,
            request.k,
            request.offset,
            request.prefilter,
            request.bypass_vector_index,
            request.nprobes,
            request.ef,
            request.refine_factor,
            request.distance_type.as_deref(),
            request.fast_search,
            request.with_row_id,
            request.lower_bound,
            request.upper_bound,
            "analyze_table_query_plan",
        )?;

        scanner.analyze_plan().await.map_err(|e| {
            Error::namespace_source(
                format!(
                    "Failed to analyze query plan for table at '{}': {}",
                    table_uri, e
                )
                .into(),
            )
        })
    }

    async fn count_table_rows(&self, request: CountTableRowsRequest) -> Result<i64> {
        self.record_op("count_table_rows");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, request.version, "count_table_rows")
            .await?;

        let count =
            dataset
                .count_rows(request.predicate)
                .await
                .map_err(|e| NamespaceError::Internal {
                    message: format!("Failed to count rows for table at '{}': {:?}", table_uri, e),
                })?;

        Ok(count as i64)
    }

    async fn insert_into_table(
        &self,
        request: InsertIntoTableRequest,
        request_data: Bytes,
    ) -> Result<InsertIntoTableResponse> {
        self.record_op("insert_into_table");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let (reader, _num_rows) =
            Self::ipc_reader_from_request_data(&request_data, "insert_into_table")?;

        let mode = match request.mode.as_deref() {
            Some(m) if m.eq_ignore_ascii_case("overwrite") => WriteMode::Overwrite,
            Some(m) if m.eq_ignore_ascii_case("append") => WriteMode::Append,
            None => WriteMode::Append,
            Some(m) => {
                return Err(lance_namespace::error::NamespaceError::InvalidInput {
                    message: format!(
                        "Unsupported write mode '{}'. Supported modes are: 'append', 'overwrite'",
                        m
                    ),
                }
                .into());
            }
        };

        if !self.table_uri_has_actual_manifests(&table_uri).await? {
            self.write_reader_to_table(&table_uri, reader, WriteMode::Create, None)
                .await?;
        } else {
            self.write_reader_to_table(&table_uri, reader, mode, None)
                .await?;
        }

        Ok(InsertIntoTableResponse {
            transaction_id: None,
        })
    }

    async fn merge_insert_into_table(
        &self,
        request: MergeInsertIntoTableRequest,
        request_data: Bytes,
    ) -> Result<MergeInsertIntoTableResponse> {
        self.record_op("merge_insert_into_table");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let on = request.on.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "'on' field is required for merge_insert_into_table".to_string(),
            })
        })?;

        let table_has_manifests = self.table_uri_has_actual_manifests(&table_uri).await?;
        let (reader, num_rows) =
            Self::ipc_reader_from_request_data(&request_data, "merge_insert_into_table")?;

        if !table_has_manifests {
            let dataset = self
                .write_reader_to_table(&table_uri, reader, WriteMode::Create, None)
                .await?;
            let version = dataset.version().version as i64;
            return Ok(MergeInsertIntoTableResponse {
                transaction_id: None,
                num_updated_rows: Some(0),
                num_inserted_rows: Some(num_rows as i64),
                num_deleted_rows: Some(0),
                version: Some(version),
            });
        }

        let dataset = Arc::new(
            self.load_dataset(&table_uri, None, "merge_insert_into_table")
                .await?,
        );

        let mut merge_builder = MergeInsertBuilder::try_new(dataset.clone(), vec![on.clone()])
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::InvalidInput {
                    message: format!("Failed to create merge_insert_into_table builder: {}", e),
                })
            })?;

        if let Some(filter) = request.when_matched_update_all_filt.as_deref() {
            let behavior = WhenMatched::update_if(dataset.as_ref(), filter).map_err(|e| {
                lance_core::Error::from(NamespaceError::InvalidInput {
                    message: format!(
                        "Invalid when_matched_update_all_filt for merge_insert_into_table: {}",
                        e
                    ),
                })
            })?;
            merge_builder.when_matched(behavior);
        } else if request.when_matched_update_all.unwrap_or(false) {
            merge_builder.when_matched(WhenMatched::UpdateAll);
        }

        if matches!(request.when_not_matched_insert_all, Some(false)) {
            merge_builder.when_not_matched(WhenNotMatched::DoNothing);
        } else {
            merge_builder.when_not_matched(WhenNotMatched::InsertAll);
        }

        if let Some(filter) = request.when_not_matched_by_source_delete_filt.as_deref() {
            let behavior = WhenNotMatchedBySource::delete_if(dataset.as_ref(), filter).map_err(|e| {
                lance_core::Error::from(NamespaceError::InvalidInput {
                    message: format!(
                        "Invalid when_not_matched_by_source_delete_filt for merge_insert_into_table: {}",
                        e
                    ),
                })
            })?;
            merge_builder.when_not_matched_by_source(behavior);
        } else if request.when_not_matched_by_source_delete.unwrap_or(false) {
            merge_builder.when_not_matched_by_source(WhenNotMatchedBySource::Delete);
        }

        if let Some(use_index) = request.use_index {
            merge_builder.use_index(use_index);
        }

        let (dataset, stats) = merge_builder
            .try_build()
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::InvalidInput {
                    message: format!("Failed to build merge_insert_into_table job: {}", e),
                })
            })?
            .execute_reader(reader)
            .await
            .map_err(|e| NamespaceError::Internal {
                message: format!(
                    "Failed to merge_insert_into_table at '{}': {}",
                    table_uri, e
                ),
            })?;

        Ok(MergeInsertIntoTableResponse {
            transaction_id: None,
            num_updated_rows: Some(stats.num_updated_rows as i64),
            num_inserted_rows: Some(stats.num_inserted_rows as i64),
            num_deleted_rows: Some(stats.num_deleted_rows as i64),
            version: Some(dataset.version().version as i64),
        })
    }

    async fn query_table(&self, request: QueryTableRequest) -> Result<Bytes> {
        use arrow::ipc::writer::FileWriter;

        self.record_op("query_table");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, request.version, "query_table")
            .await?;

        // Build scanner
        let mut scanner = dataset.scan();

        // Check if this is a vector search query
        // vector is Box<QueryTableRequestVector>, not Option
        let has_vector_query = request
            .vector
            .single_vector
            .as_ref()
            .map(|sv| !sv.is_empty())
            .unwrap_or(false)
            || request
                .vector
                .multi_vector
                .as_ref()
                .map(|mv| !mv.is_empty())
                .unwrap_or(false);

        // Apply prefilter setting (must be set before nearest)
        if let Some(prefilter) = request.prefilter {
            scanner.prefilter(prefilter);
        }

        // Apply vector search if query vector is provided
        if has_vector_query {
            let vector_column = request.vector_column.as_deref().unwrap_or("vector");

            // Get the query vector(s)
            let query_vector: Vec<f32> = request
                .vector
                .single_vector
                .clone()
                .or_else(|| {
                    request
                        .vector
                        .multi_vector
                        .as_ref()
                        .and_then(|mv| mv.first().cloned())
                })
                .unwrap_or_default();

            if !query_vector.is_empty() {
                let k = if request.k > 0 {
                    request.k as usize
                } else {
                    10
                };
                let query_array = Float32Array::from(query_vector);
                scanner
                    .nearest(vector_column, &query_array, k)
                    .map_err(|e| NamespaceError::InvalidInput {
                        message: format!("Invalid vector search: {:?}", e),
                    })?;

                // Apply distance type if specified
                if let Some(ref distance_type) = request.distance_type {
                    let metric = match distance_type.to_lowercase().as_str() {
                        "l2" | "euclidean" => MetricType::L2,
                        "cosine" => MetricType::Cosine,
                        "dot" | "inner_product" => MetricType::Dot,
                        "hamming" => MetricType::Hamming,
                        _ => {
                            return Err(NamespaceError::InvalidInput {
                                message: format!("Unknown distance type: {}", distance_type),
                            }
                            .into());
                        }
                    };
                    scanner.distance_metric(metric);
                }

                // Apply nprobes if specified (maps to minimum_nprobes, matching lancedb behavior)
                if let Some(nprobes) = request.nprobes {
                    scanner.minimum_nprobes(nprobes as usize);
                }

                // Apply ef (HNSW search effort) if specified
                if let Some(ef) = request.ef {
                    scanner.ef(ef as usize);
                }

                // Apply refine_factor if specified
                if let Some(refine_factor) = request.refine_factor {
                    scanner.refine(refine_factor as u32);
                }

                // Apply distance bounds if specified
                if request.lower_bound.is_some() || request.upper_bound.is_some() {
                    scanner.distance_range(request.lower_bound, request.upper_bound);
                }

                // Apply use_index (inverse of bypass_vector_index)
                if let Some(bypass) = request.bypass_vector_index {
                    scanner.use_index(!bypass);
                }

                // Apply fast_search if specified
                if request.fast_search == Some(true) {
                    scanner.fast_search();
                }
            }
        }

        // Apply full text search if specified
        if let Some(ref fts_query) = request.full_text_query {
            // Handle string_query (simple string FTS)
            if let Some(ref string_query) = fts_query.string_query {
                let mut fts = FullTextSearchQuery::new(string_query.query.clone());

                // Apply column filter if specified
                if let Some(ref columns) = string_query.columns
                    && !columns.is_empty()
                {
                    fts = fts
                        .with_columns(columns)
                        .map_err(|e| NamespaceError::InvalidInput {
                            message: format!("Invalid FTS columns: {:?}", e),
                        })?;
                }

                scanner
                    .full_text_search(fts)
                    .map_err(|e| NamespaceError::InvalidInput {
                        message: format!("Invalid full text search: {:?}", e),
                    })?;
            }
            // Note: structured_query would require more complex parsing
            // For now, we only support string_query
        }

        // Apply column projection if specified
        if let Some(ref columns) = request.columns {
            if let Some(ref column_names) = columns.column_names
                && !column_names.is_empty()
            {
                scanner
                    .project(column_names)
                    .map_err(|e| NamespaceError::InvalidInput {
                        message: format!("Invalid column projection: {:?}", e),
                    })?;
            } else if let Some(ref column_aliases) = columns.column_aliases
                && !column_aliases.is_empty()
            {
                // column_aliases is HashMap<String, String> where key is alias, value is SQL expression
                let transform_pairs: Vec<(String, String)> = column_aliases
                    .iter()
                    .map(|(alias, sql)| (alias.clone(), sql.clone()))
                    .collect();
                scanner
                    .project_with_transform(
                        &transform_pairs
                            .iter()
                            .map(|(a, s)| (a.as_str(), s.as_str()))
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|e| NamespaceError::InvalidInput {
                        message: format!("Invalid column alias expression: {:?}", e),
                    })?;
            }
        }

        // Apply filter if specified
        if let Some(ref filter) = request.filter
            && !filter.is_empty()
        {
            scanner
                .filter(filter)
                .map_err(|e| NamespaceError::InvalidInput {
                    message: format!("Invalid filter expression: {:?}", e),
                })?;
        }

        // Apply with_row_id if requested
        if request.with_row_id == Some(true) {
            scanner.with_row_id();
        }

        // Apply limit if specified (k is the number of results to return)
        // k == 0 means no limit
        // Note: For vector search, limit is already applied via nearest()
        if !has_vector_query && request.k > 0 {
            let offset = request.offset.map(|o| o as i64);
            scanner.limit(Some(request.k as i64), offset).map_err(|e| {
                NamespaceError::InvalidInput {
                    message: format!("Invalid limit/offset: {:?}", e),
                }
            })?;
        } else if has_vector_query && request.offset.is_some() {
            // For vector search, offset is handled separately
            let offset = request.offset.map(|o| o as i64);
            scanner
                .limit(None, offset)
                .map_err(|e| NamespaceError::InvalidInput {
                    message: format!("Invalid offset: {:?}", e),
                })?;
        }

        // Execute the scan and collect results
        let batch = scanner
            .try_into_batch()
            .await
            .map_err(|e| NamespaceError::Internal {
                message: format!("Failed to execute query: {:?}", e),
            })?;

        // Serialize to Arrow IPC file format
        let schema = batch.schema();
        let mut buffer = Vec::new();
        {
            let mut writer = FileWriter::try_new(&mut buffer, &schema).map_err(|e| {
                NamespaceError::Internal {
                    message: format!("Failed to create IPC writer: {:?}", e),
                }
            })?;
            writer.write(&batch).map_err(|e| NamespaceError::Internal {
                message: format!("Failed to write batch to IPC: {:?}", e),
            })?;
            writer.finish().map_err(|e| NamespaceError::Internal {
                message: format!("Failed to finish IPC writer: {:?}", e),
            })?;
        }

        Ok(Bytes::from(buffer))
    }

    async fn list_table_tags(
        &self,
        request: ListTableTagsRequest,
    ) -> Result<ListTableTagsResponse> {
        self.record_op("list_table_tags");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, None, "list_table_tags")
            .await?;

        let raw_tags = dataset.tags().list().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to list tags for table at '{}': {}", table_uri, e),
            })
        })?;

        let tags = raw_tags
            .into_iter()
            .map(|(name, contents)| {
                let mut tag_model =
                    ModelTagContents::new(contents.version as i64, contents.manifest_size as i64);
                tag_model.branch = contents.branch;
                (name, tag_model)
            })
            .collect();

        Ok(ListTableTagsResponse {
            tags,
            page_token: None,
        })
    }

    async fn get_table_tag_version(
        &self,
        request: GetTableTagVersionRequest,
    ) -> Result<GetTableTagVersionResponse> {
        self.record_op("get_table_tag_version");
        if request.tag.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "tag name must not be empty for get_table_tag_version".to_string(),
            }
            .into());
        }

        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, None, "get_table_tag_version")
            .await?;

        let version = dataset
            .tags()
            .get_version(&request.tag)
            .await
            .map_err(|e| Self::map_tag_error(e, &request.tag, &table_uri))?;

        Ok(GetTableTagVersionResponse {
            version: version as i64,
        })
    }

    async fn create_table_tag(
        &self,
        request: CreateTableTagRequest,
    ) -> Result<CreateTableTagResponse> {
        self.record_op("create_table_tag");
        if request.tag.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "tag name must not be empty for create_table_tag".to_string(),
            }
            .into());
        }
        if request.version <= 0 {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "tag version must be a positive integer, got {} for create_table_tag",
                    request.version
                ),
            }
            .into());
        }

        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, None, "create_table_tag")
            .await?;

        dataset
            .tags()
            .create(&request.tag, request.version as u64)
            .await
            .map_err(|e| Self::map_tag_error(e, &request.tag, &table_uri))?;

        Ok(CreateTableTagResponse {
            transaction_id: None,
        })
    }

    async fn delete_table_tag(
        &self,
        request: DeleteTableTagRequest,
    ) -> Result<DeleteTableTagResponse> {
        self.record_op("delete_table_tag");
        if request.tag.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "tag name must not be empty for delete_table_tag".to_string(),
            }
            .into());
        }

        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, None, "delete_table_tag")
            .await?;

        dataset
            .tags()
            .delete(&request.tag)
            .await
            .map_err(|e| Self::map_tag_error(e, &request.tag, &table_uri))?;

        Ok(DeleteTableTagResponse {
            transaction_id: None,
        })
    }

    async fn update_table_tag(
        &self,
        request: UpdateTableTagRequest,
    ) -> Result<UpdateTableTagResponse> {
        self.record_op("update_table_tag");
        if request.tag.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "tag name must not be empty for update_table_tag".to_string(),
            }
            .into());
        }
        if request.version <= 0 {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "tag version must be a positive integer, got {} for update_table_tag",
                    request.version
                ),
            }
            .into());
        }

        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, None, "update_table_tag")
            .await?;

        dataset
            .tags()
            .update(&request.tag, request.version as u64)
            .await
            .map_err(|e| Self::map_tag_error(e, &request.tag, &table_uri))?;

        Ok(UpdateTableTagResponse {
            transaction_id: None,
        })
    }

    fn namespace_id(&self) -> String {
        format!("DirectoryNamespace {{ root: {:?} }}", self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_ipc::reader::{FileReader, StreamReader};
    use lance::dataset::Dataset;
    use lance::index::DatasetIndexExt;
    use lance_core::utils::tempfile::{TempStdDir, TempStrDir};
    use lance_core::utils::testing::CountingObjectStore;
    use lance_io::object_store::{providers::local::FileStoreProvider, uri_to_url};
    use lance_namespace::models::{
        CreateTableRequest, JsonArrowDataType, JsonArrowField, JsonArrowSchema, ListTablesRequest,
        QueryTableRequestColumns,
    };
    use lance_namespace::schema::convert_json_arrow_schema;
    use std::io::Cursor;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use url::Url;

    fn assert_plan_contains_all(plan: &str, expected_fragments: &[&str], context: &str) {
        for expected_fragment in expected_fragments {
            assert!(
                plan.contains(expected_fragment),
                "{}. Missing fragment: '{}'. Plan:\n{}",
                context,
                expected_fragment,
                plan
            );
        }
    }

    /// Helper to create a test DirectoryNamespace with a temporary directory
    async fn create_test_namespace() -> (DirectoryNamespace, TempStdDir) {
        let temp_dir = TempStdDir::default();

        let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
            .build()
            .await
            .unwrap();
        (namespace, temp_dir)
    }

    #[derive(Debug)]
    struct CountingFileStoreProvider {
        listing_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl lance_io::object_store::ObjectStoreProvider for CountingFileStoreProvider {
        async fn new_store(
            &self,
            base_path: Url,
            params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            let provider = FileStoreProvider;
            let mut store = provider.new_store(base_path, params).await?;
            store.inner = Arc::new(CountingObjectStore::new(
                store.inner.clone(),
                self.listing_count.clone(),
            ));
            Ok(store)
        }

        fn extract_path(&self, url: &Url) -> Result<Path> {
            let provider = FileStoreProvider;
            provider.extract_path(url)
        }

        fn calculate_object_store_prefix(
            &self,
            url: &Url,
            storage_options: Option<&HashMap<String, String>>,
        ) -> Result<String> {
            let provider = FileStoreProvider;
            provider.calculate_object_store_prefix(url, storage_options)
        }
    }

    fn file_object_store_uri(path: &str) -> String {
        let file_url = uri_to_url(path).unwrap();
        let mut url = Url::parse("file-object-store:///").unwrap();
        url.set_path(file_url.path());
        url.to_string()
    }

    fn build_listing_counting_session(listing_count: Arc<AtomicUsize>) -> Arc<Session> {
        let registry = Arc::new(ObjectStoreRegistry::default());
        registry.insert(
            "file-object-store",
            Arc::new(CountingFileStoreProvider { listing_count }),
        );
        Arc::new(Session::new(0, 0, registry))
    }

    /// Helper to create test IPC data from a schema
    fn create_test_ipc_data(schema: &JsonArrowSchema) -> Vec<u8> {
        use arrow::ipc::writer::StreamWriter;

        let arrow_schema = convert_json_arrow_schema(schema).unwrap();
        let arrow_schema = Arc::new(arrow_schema);
        let batch = arrow::record_batch::RecordBatch::new_empty(arrow_schema.clone());
        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        buffer
    }

    fn create_ipc_data_from_batches(
        schema: Arc<arrow_schema::Schema>,
        batches: Vec<arrow::record_batch::RecordBatch>,
    ) -> Vec<u8> {
        use arrow::ipc::writer::StreamWriter;

        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
            for batch in &batches {
                writer.write(batch).unwrap();
            }
            writer.finish().unwrap();
        }
        buffer
    }

    fn create_non_empty_test_ipc_data() -> Vec<u8> {
        use arrow::array::{Int32Array, StringArray};
        use arrow::record_batch::RecordBatch;

        let schema = Arc::new(convert_json_arrow_schema(&create_test_schema()).unwrap());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec![Some("alice"), Some("bob")])),
            ],
        )
        .unwrap();
        create_ipc_data_from_batches(schema, vec![batch])
    }

    fn create_single_row_test_ipc_data() -> Vec<u8> {
        use arrow::array::{Int32Array, StringArray};
        use arrow::record_batch::RecordBatch;

        let schema = Arc::new(convert_json_arrow_schema(&create_test_schema()).unwrap());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![10])),
                Arc::new(StringArray::from(vec![Some("carol")])),
            ],
        )
        .unwrap();
        create_ipc_data_from_batches(schema, vec![batch])
    }

    /// Helper to create a simple test schema
    fn create_test_schema() -> JsonArrowSchema {
        let int_type = JsonArrowDataType::new("int32".to_string());
        let string_type = JsonArrowDataType::new("utf8".to_string());

        let id_field = JsonArrowField {
            name: "id".to_string(),
            r#type: Box::new(int_type),
            nullable: false,
            metadata: None,
        };

        let name_field = JsonArrowField {
            name: "name".to_string(),
            r#type: Box::new(string_type),
            nullable: true,
            metadata: None,
        };

        JsonArrowSchema {
            fields: vec![id_field, name_field],
            metadata: None,
        }
    }

    fn create_scalar_table_ipc_data() -> Vec<u8> {
        use arrow::array::{Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["alice", "bob", "cory"])),
            ],
        )
        .unwrap();
        create_ipc_data_from_batches(schema, vec![batch])
    }

    fn create_vector_table_ipc_data() -> Vec<u8> {
        use arrow::array::{FixedSizeListArray, Float32Array, Int32Array};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                true,
            ),
        ]));
        let vector_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vectors = FixedSizeListArray::try_new(
            vector_field,
            2,
            Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
            None,
        )
        .unwrap();
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(vectors)],
        )
        .unwrap();
        create_ipc_data_from_batches(schema, vec![batch])
    }

    async fn create_scalar_table(namespace: &DirectoryNamespace, table_name: &str) {
        let mut create_table_request = CreateTableRequest::new();
        create_table_request.id = Some(vec![table_name.to_string()]);
        namespace
            .create_table(
                create_table_request,
                Bytes::from(create_scalar_table_ipc_data()),
            )
            .await
            .unwrap();
    }

    async fn create_vector_table(namespace: &DirectoryNamespace, table_name: &str) {
        let mut create_table_request = CreateTableRequest::new();
        create_table_request.id = Some(vec![table_name.to_string()]);
        namespace
            .create_table(
                create_table_request,
                Bytes::from(create_vector_table_ipc_data()),
            )
            .await
            .unwrap();
    }

    async fn create_sharded_namespace(
        temp_path: &str,
        catalog_branch: Option<&str>,
    ) -> DirectoryNamespace {
        let mut builder = DirectoryNamespaceBuilder::new(temp_path)
            .dir_listing_enabled(false)
            .manifest_shard_count(4);
        if let Some(branch) = catalog_branch {
            builder = builder.catalog_branch(branch);
        }
        builder.build().await.unwrap()
    }

    #[tokio::test]
    async fn test_sharded_manifest_table_operations() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let namespace = create_sharded_namespace(temp_path, None).await;

        create_scalar_table(&namespace, "alpha").await;
        create_scalar_table(&namespace, "beta").await;

        let response = namespace
            .list_tables(ListTablesRequest {
                id: Some(vec![]),
                page_token: None,
                limit: None,
                ..Default::default()
            })
            .await
            .unwrap();
        let tables = response.tables.into_iter().collect::<HashSet<_>>();
        assert_eq!(
            tables,
            HashSet::from(["alpha".to_string(), "beta".to_string()])
        );

        let describe = namespace
            .describe_table(DescribeTableRequest {
                id: Some(vec!["alpha".to_string()]),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(describe.table.as_deref(), Some("alpha"));

        for shard_index in 0..4 {
            let shard_uri = format!("{}/__manifest_shard_{:06}", temp_path, shard_index);
            DatasetBuilder::from_uri(shard_uri).load().await.unwrap();
        }
        DatasetBuilder::from_uri(format!("{}/__super_manifest", temp_path))
            .load()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_sharded_catalog_branch_promotion() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let main = create_sharded_namespace(temp_path, None).await;

        create_scalar_table(&main, "main_table").await;
        main.create_catalog_branch("experiment").await.unwrap();
        main.create_catalog_branch("stale").await.unwrap();

        let experiment = create_sharded_namespace(temp_path, Some("experiment")).await;
        create_scalar_table(&experiment, "branch_table").await;

        let main_tables = main
            .list_tables(ListTablesRequest {
                id: Some(vec![]),
                page_token: None,
                limit: None,
                ..Default::default()
            })
            .await
            .unwrap()
            .tables
            .into_iter()
            .collect::<HashSet<_>>();
        assert!(main_tables.contains("main_table"));
        assert!(!main_tables.contains("branch_table"));

        let branch_tables = experiment
            .list_tables(ListTablesRequest {
                id: Some(vec![]),
                page_token: None,
                limit: None,
                ..Default::default()
            })
            .await
            .unwrap()
            .tables
            .into_iter()
            .collect::<HashSet<_>>();
        assert!(branch_tables.contains("main_table"));
        assert!(branch_tables.contains("branch_table"));

        main.promote_catalog_branch("experiment", "main")
            .await
            .unwrap();
        assert!(main.promote_catalog_branch("stale", "main").await.is_err());

        let reopened_main = create_sharded_namespace(temp_path, None).await;
        let promoted_tables = reopened_main
            .list_tables(ListTablesRequest {
                id: Some(vec![]),
                page_token: None,
                limit: None,
                ..Default::default()
            })
            .await
            .unwrap()
            .tables
            .into_iter()
            .collect::<HashSet<_>>();
        assert!(promoted_tables.contains("main_table"));
        assert!(promoted_tables.contains("branch_table"));
    }

    #[tokio::test]
    async fn test_sharded_catalog_branch_table_writes_are_isolated() {
        use arrow::array::{Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use lance::dataset::{WriteMode, WriteParams};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let main = create_sharded_namespace(temp_path, None).await;

        create_scalar_table(&main, "isolated").await;
        assert_eq!(
            open_dataset(&main, "isolated")
                .await
                .count_rows(None)
                .await
                .unwrap(),
            3
        );
        let main_location = main
            .describe_table(DescribeTableRequest {
                id: Some(vec!["isolated".to_string()]),
                ..Default::default()
            })
            .await
            .unwrap()
            .location
            .unwrap();

        main.create_catalog_branch("experiment").await.unwrap();
        let experiment = create_sharded_namespace(temp_path, Some("experiment")).await;
        let branch_location = experiment
            .describe_table(DescribeTableRequest {
                id: Some(vec!["isolated".to_string()]),
                ..Default::default()
            })
            .await
            .unwrap()
            .location
            .unwrap();
        assert_ne!(main_location, branch_location);
        assert!(branch_location.contains("/tree/experiment"));

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![4])),
                Arc::new(StringArray::from(vec!["dora"])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let experiment_ns = Arc::new(experiment) as Arc<dyn LanceNamespace>;
        Dataset::write_into_namespace(
            reader,
            experiment_ns,
            vec!["isolated".to_string()],
            Some(write_params),
        )
        .await
        .unwrap();

        assert_eq!(
            open_dataset(&main, "isolated")
                .await
                .count_rows(None)
                .await
                .unwrap(),
            3
        );
        let reopened_experiment = create_sharded_namespace(temp_path, Some("experiment")).await;
        assert_eq!(
            open_dataset(&reopened_experiment, "isolated")
                .await
                .count_rows(None)
                .await
                .unwrap(),
            4
        );

        main.promote_catalog_branch("experiment", "main")
            .await
            .unwrap();
        let reopened_main = create_sharded_namespace(temp_path, None).await;
        assert_eq!(
            open_dataset(&reopened_main, "isolated")
                .await
                .count_rows(None)
                .await
                .unwrap(),
            4
        );
    }

    #[tokio::test]
    async fn test_sharded_table_version_storage_records_versions() {
        use futures::TryStreamExt;
        use lance_namespace::models::{CreateTableVersionRequest, ListTableVersionsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .dir_listing_enabled(false)
            .manifest_shard_count(4)
            .table_version_tracking_enabled(true)
            .table_version_storage_enabled(true)
            .build()
            .await
            .unwrap();

        create_scalar_table(&namespace, "versions").await;
        let describe = namespace
            .describe_table(DescribeTableRequest {
                id: Some(vec!["versions".to_string()]),
                ..Default::default()
            })
            .await
            .unwrap();
        let table_uri = describe.location.unwrap();
        let dataset = Dataset::open(&table_uri).await.unwrap();
        let object_store = dataset.object_store(None).await.unwrap();
        let manifest_metas: Vec<_> = object_store
            .inner
            .list(Some(&dataset.versions_dir()))
            .try_collect()
            .await
            .unwrap();
        let manifest_meta = manifest_metas
            .iter()
            .find(|meta| {
                meta.location
                    .filename()
                    .is_some_and(|filename| filename.ends_with(".manifest"))
            })
            .unwrap();
        let manifest_data = object_store
            .inner
            .get(&manifest_meta.location)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let staging_path = dataset.versions_dir().join("staging_manifest");
        object_store
            .inner
            .put(&staging_path, manifest_data.into())
            .await
            .unwrap();

        let mut create_req = CreateTableVersionRequest::new(2, staging_path.to_string());
        create_req.id = Some(vec!["versions".to_string()]);
        create_req.naming_scheme = Some("V2".to_string());
        namespace.create_table_version(create_req).await.unwrap();

        let mut list_req = ListTableVersionsRequest::new();
        list_req.id = Some(vec!["versions".to_string()]);
        let versions = namespace.list_table_versions(list_req).await.unwrap();
        assert_eq!(versions.versions.len(), 1);
        assert_eq!(versions.versions[0].version, 2);

        let describe_version = namespace
            .describe_table_version(DescribeTableVersionRequest {
                id: Some(vec!["versions".to_string()]),
                version: Some(2),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(describe_version.version.version, 2);
    }

    async fn open_dataset(namespace: &DirectoryNamespace, table_name: &str) -> Dataset {
        let mut describe_request = DescribeTableRequest::new();
        describe_request.id = Some(vec![table_name.to_string()]);
        let table_uri = namespace
            .describe_table(describe_request)
            .await
            .unwrap()
            .location
            .expect("table location should exist");
        Dataset::open(&table_uri).await.unwrap()
    }

    async fn create_scalar_index(
        namespace: &DirectoryNamespace,
        table_name: &str,
        index_name: &str,
    ) -> Option<String> {
        use lance_namespace::models::CreateTableIndexRequest;

        let mut create_index_request =
            CreateTableIndexRequest::new("id".to_string(), "BTREE".to_string());
        create_index_request.id = Some(vec![table_name.to_string()]);
        create_index_request.name = Some(index_name.to_string());
        namespace
            .create_table_scalar_index(create_index_request)
            .await
            .unwrap()
            .transaction_id
    }

    #[tokio::test]
    async fn test_create_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.is_some());
        assert!(response.location.unwrap().ends_with("test_table.lance"));
        assert_eq!(response.version, Some(1));
    }

    #[tokio::test]
    async fn test_create_table_without_data() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let result = namespace.create_table(request, bytes::Bytes::new()).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Arrow IPC stream) is required")
        );
    }

    #[tokio::test]
    async fn test_create_table_with_invalid_id() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Test with empty ID
        let mut request = CreateTableRequest::new();
        request.id = Some(vec![]);

        let result = namespace
            .create_table(request, bytes::Bytes::from(ipc_data.clone()))
            .await;
        assert!(result.is_err());

        // Test with multi-level ID - should now work with manifest enabled
        // First create the parent namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_namespace".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Now create table in the namespace
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_namespace".to_string(), "table".to_string()]);

        let result = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await;
        // Should succeed with manifest enabled
        assert!(
            result.is_ok(),
            "Multi-level table IDs should work with manifest enabled"
        );
    }

    #[tokio::test]
    async fn test_list_tables() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Initially, no tables
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["table1".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data.clone()))
            .await
            .unwrap();

        // Create another table
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["table2".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // List tables should return both
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = namespace.list_tables(request).await.unwrap();
        let tables = response.tables;
        assert_eq!(tables.len(), 2);
        assert!(tables.contains(&"table1".to_string()));
        assert!(tables.contains(&"table2".to_string()));
    }

    #[tokio::test]
    async fn test_list_tables_pagination() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        for name in ["alpha", "bravo", "charlie"] {
            let mut req = CreateTableRequest::new();
            req.id = Some(vec![name.to_string()]);
            namespace
                .create_table(req, bytes::Bytes::from(ipc_data.clone()))
                .await
                .unwrap();
        }

        // First page: limit=2, no page_token
        let first_page = namespace
            .list_tables(ListTablesRequest {
                id: Some(vec![]),
                limit: Some(2),
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(first_page.tables, vec!["alpha", "bravo"]);
        assert_eq!(first_page.page_token.as_deref(), Some("bravo"));

        // Second page: use page_token from first response
        let second_page = namespace
            .list_tables(ListTablesRequest {
                id: Some(vec![]),
                limit: Some(2),
                page_token: first_page.page_token.clone(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(second_page.tables, vec!["charlie"]);
        assert!(second_page.page_token.is_none());
    }

    #[tokio::test]
    async fn test_list_tables_pagination_limit_zero() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut req = CreateTableRequest::new();
        req.id = Some(vec!["alpha".to_string()]);
        namespace
            .create_table(req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        let response = namespace
            .list_tables(ListTablesRequest {
                id: Some(vec![]),
                limit: Some(0),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response.tables.is_empty());
        assert!(response.page_token.is_none());
    }

    #[tokio::test]
    async fn test_list_tables_with_namespace_id() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // First create a child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_namespace".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Now list tables in the child namespace
        let mut request = ListTablesRequest::new();
        request.id = Some(vec!["test_namespace".to_string()]);

        let result = namespace.list_tables(request).await;
        // Should succeed (with manifest enabled) and return empty list (no tables yet)
        assert!(
            result.is_ok(),
            "list_tables should work with child namespace when manifest is enabled"
        );
        let response = result.unwrap();
        assert_eq!(
            response.tables.len(),
            0,
            "Namespace should have no tables yet"
        );
    }

    #[tokio::test]
    async fn test_create_scalar_index() {
        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "users").await;

        let transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;
        let dataset = open_dataset(&namespace, "users").await;
        let expected_transaction_id = dataset
            .read_transaction()
            .await
            .unwrap()
            .map(|transaction| transaction.uuid);
        assert_eq!(transaction_id, expected_transaction_id);
        let indices = dataset.load_indices().await.unwrap();
        assert!(indices.iter().any(|index| index.name == "users_id_idx"));
    }

    #[tokio::test]
    async fn test_create_vector_index() {
        use lance_namespace::models::CreateTableIndexRequest;

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_vector_table(&namespace, "vectors").await;

        let mut create_index_request =
            CreateTableIndexRequest::new("vector".to_string(), "IVF_FLAT".to_string());
        create_index_request.id = Some(vec!["vectors".to_string()]);
        create_index_request.name = Some("vector_idx".to_string());
        create_index_request.distance_type = Some("l2".to_string());
        let transaction_id = namespace
            .create_table_index(create_index_request)
            .await
            .unwrap()
            .transaction_id;

        let dataset = open_dataset(&namespace, "vectors").await;
        let expected_transaction_id = dataset
            .read_transaction()
            .await
            .unwrap()
            .map(|transaction| transaction.uuid);
        assert_eq!(transaction_id, expected_transaction_id);
        let indices = dataset.load_indices().await.unwrap();
        assert!(indices.iter().any(|index| index.name == "vector_idx"));
    }

    #[tokio::test]
    async fn test_list_table_indices() {
        use lance_namespace::models::ListTableIndicesRequest;

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "users").await;
        create_scalar_index(&namespace, "users", "a_idx").await;
        create_scalar_index(&namespace, "users", "b_idx").await;
        let transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;

        let response = namespace
            .list_table_indices(ListTableIndicesRequest {
                id: Some(vec!["users".to_string()]),
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(response.indexes.len(), 3);
        assert_eq!(response.indexes[0].index_name, "a_idx");
        assert_eq!(response.indexes[1].index_name, "b_idx");
        assert_eq!(response.indexes[2].index_name, "users_id_idx");
        assert!(response.page_token.is_none());
        let users_id_idx = response
            .indexes
            .iter()
            .find(|index| index.index_name == "users_id_idx")
            .unwrap();
        assert_eq!(users_id_idx.columns, vec!["id"]);
        assert_eq!(users_id_idx.status, "SUCCEEDED");

        let dataset = open_dataset(&namespace, "users").await;
        let expected_transaction_id = dataset
            .read_transaction()
            .await
            .unwrap()
            .map(|transaction| transaction.uuid);
        assert_eq!(transaction_id, expected_transaction_id);
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices
                .iter()
                .filter(|index| index.name == "users_id_idx")
                .count(),
            1
        );

        let first_page = namespace
            .list_table_indices(ListTableIndicesRequest {
                id: Some(vec!["users".to_string()]),
                limit: Some(2),
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(first_page.indexes.len(), 2);
        assert_eq!(first_page.indexes[0].index_name, "a_idx");
        assert_eq!(first_page.indexes[1].index_name, "b_idx");
        assert_eq!(first_page.page_token.as_deref(), Some("b_idx"));

        let second_page = namespace
            .list_table_indices(ListTableIndicesRequest {
                id: Some(vec!["users".to_string()]),
                page_token: first_page.page_token.clone(),
                limit: Some(2),
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(second_page.indexes.len(), 1);
        assert_eq!(second_page.indexes[0].index_name, "users_id_idx");
        assert!(second_page.page_token.is_none());
    }

    #[tokio::test]
    async fn test_describe_table_index_stats() {
        use lance_namespace::models::DescribeTableIndexStatsRequest;

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "users").await;
        let transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;

        let response = namespace
            .describe_table_index_stats(DescribeTableIndexStatsRequest {
                id: Some(vec!["users".to_string()]),
                index_name: Some("users_id_idx".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(response.index_type, Some("BTree".to_string()));
        assert_eq!(response.num_indices, Some(1));
        assert_eq!(response.num_indexed_rows, Some(3));
        assert_eq!(response.num_unindexed_rows, Some(0));

        let dataset = open_dataset(&namespace, "users").await;
        let expected_transaction_id = dataset
            .read_transaction()
            .await
            .unwrap()
            .map(|transaction| transaction.uuid);
        assert_eq!(transaction_id, expected_transaction_id);
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("users_id_idx").await.unwrap()).unwrap();
        assert_eq!(stats["index_type"], "BTree");
        assert_eq!(stats["num_indices"], 1);
        assert_eq!(stats["num_indexed_rows"], 3);
        assert_eq!(stats["num_unindexed_rows"], 0);
    }

    #[tokio::test]
    async fn test_describe_transaction() {
        use lance_namespace::models::DescribeTransactionRequest;

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "users").await;
        let transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;
        let dataset = open_dataset(&namespace, "users").await;
        let latest_transaction = dataset.read_transaction().await.unwrap();
        assert_eq!(
            transaction_id,
            latest_transaction
                .as_ref()
                .map(|transaction| transaction.uuid.clone())
        );

        if let Some(transaction_id) = transaction_id {
            let response = namespace
                .describe_transaction(DescribeTransactionRequest {
                    id: Some(vec!["users".to_string(), transaction_id.clone()]),
                    ..Default::default()
                })
                .await
                .unwrap();
            assert_eq!(response.status, "SUCCEEDED");
            assert_eq!(
                response
                    .properties
                    .as_ref()
                    .and_then(|props| props.get("operation")),
                Some(&"CreateIndex".to_string())
            );
            assert_eq!(
                response
                    .properties
                    .as_ref()
                    .and_then(|props| props.get("uuid")),
                Some(&transaction_id)
            );
        } else {
            assert!(latest_transaction.is_none());
        }
    }

    #[tokio::test]
    async fn test_drop_table_index() {
        use lance_namespace::models::{DropTableIndexRequest, ListTableIndicesRequest};

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "users").await;
        let create_transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;

        let drop_transaction_id = namespace
            .drop_table_index(DropTableIndexRequest {
                id: Some(vec!["users".to_string()]),
                index_name: Some("users_id_idx".to_string()),
                ..Default::default()
            })
            .await
            .unwrap()
            .transaction_id;

        let dataset = open_dataset(&namespace, "users").await;
        let previous_dataset = dataset
            .checkout_version(dataset.version().version - 1)
            .await
            .unwrap();
        let previous_transaction_id = previous_dataset
            .read_transaction()
            .await
            .unwrap()
            .map(|transaction| transaction.uuid);
        assert_eq!(create_transaction_id, previous_transaction_id);
        let expected_drop_transaction_id = dataset
            .read_transaction()
            .await
            .unwrap()
            .map(|transaction| transaction.uuid);
        assert_eq!(drop_transaction_id, expected_drop_transaction_id);
        let indices = dataset.load_indices().await.unwrap();
        assert!(!indices.iter().any(|index| index.name == "users_id_idx"));

        let list_response = namespace
            .list_table_indices(ListTableIndicesRequest {
                id: Some(vec!["users".to_string()]),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(list_response.indexes.is_empty());
    }

    #[tokio::test]
    async fn test_describe_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create a table first
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Describe the table
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);
        let response = namespace.describe_table(request).await.unwrap();

        assert!(response.location.is_some());
        assert!(response.location.unwrap().ends_with("test_table.lance"));
    }

    #[tokio::test]
    async fn test_describe_nonexistent_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);

        let result = namespace.describe_table(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Table not found"));
    }

    #[tokio::test]
    async fn test_table_exists() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["existing_table".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Check existing table
        let mut request = TableExistsRequest::new();
        request.id = Some(vec!["existing_table".to_string()]);
        let result = namespace.table_exists(request).await;
        assert!(result.is_ok());

        // Check non-existent table
        let mut request = TableExistsRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);
        let result = namespace.table_exists(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Table not found"));
    }

    #[tokio::test]
    async fn test_drop_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["table_to_drop".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Verify it exists
        let mut exists_request = TableExistsRequest::new();
        exists_request.id = Some(vec!["table_to_drop".to_string()]);
        assert!(namespace.table_exists(exists_request.clone()).await.is_ok());

        // Drop the table
        let mut drop_request = DropTableRequest::new();
        drop_request.id = Some(vec!["table_to_drop".to_string()]);
        let response = namespace.drop_table(drop_request).await.unwrap();
        assert!(response.location.is_some());

        // Verify it no longer exists
        assert!(namespace.table_exists(exists_request).await.is_err());
    }

    #[tokio::test]
    async fn test_drop_nonexistent_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let mut request = DropTableRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);

        // Should not fail when dropping non-existent table (idempotent)
        let result = namespace.drop_table(request).await;
        // The operation might succeed or fail depending on implementation
        // But it should not panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_root_namespace_operations() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Test list_namespaces - should return empty list for root
        let mut request = ListNamespacesRequest::new();
        request.id = Some(vec![]);
        let result = namespace.list_namespaces(request).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().namespaces.len(), 0);

        // Test describe_namespace - should succeed for root
        let mut request = DescribeNamespaceRequest::new();
        request.id = Some(vec![]);
        let result = namespace.describe_namespace(request).await;
        assert!(result.is_ok());

        // Test namespace_exists - root always exists
        let mut request = NamespaceExistsRequest::new();
        request.id = Some(vec![]);
        let result = namespace.namespace_exists(request).await;
        assert!(result.is_ok());

        // Test create_namespace - root cannot be created
        let mut request = CreateNamespaceRequest::new();
        request.id = Some(vec![]);
        let result = namespace.create_namespace(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));

        // Test drop_namespace - root cannot be dropped
        let mut request = DropNamespaceRequest::new();
        request.id = Some(vec![]);
        let result = namespace.drop_namespace(request).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("cannot be dropped")
        );
    }

    #[tokio::test]
    async fn test_non_root_namespace_operations() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // With manifest enabled (default), child namespaces are now supported
        // Test create_namespace for non-root - should succeed with manifest
        let mut request = CreateNamespaceRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.create_namespace(request).await;
        assert!(
            result.is_ok(),
            "Child namespace creation should succeed with manifest enabled"
        );

        // Test namespace_exists for non-root - should exist after creation
        let mut request = NamespaceExistsRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.namespace_exists(request).await;
        assert!(
            result.is_ok(),
            "Child namespace should exist after creation"
        );

        // Test drop_namespace for non-root - should succeed
        let mut request = DropNamespaceRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.drop_namespace(request).await;
        assert!(
            result.is_ok(),
            "Child namespace drop should succeed with manifest enabled"
        );

        // Verify namespace no longer exists
        let mut request = NamespaceExistsRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.namespace_exists(request).await;
        assert!(
            result.is_err(),
            "Child namespace should not exist after drop"
        );
    }

    #[tokio::test]
    async fn test_config_custom_root() {
        let temp_dir = TempStdDir::default();
        let custom_path = temp_dir.join("custom");
        std::fs::create_dir(&custom_path).unwrap();

        let namespace = DirectoryNamespaceBuilder::new(custom_path.to_string_lossy().to_string())
            .build()
            .await
            .unwrap();

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table and verify location
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.unwrap().contains("custom"));
    }

    #[tokio::test]
    async fn test_config_storage_options() {
        let temp_dir = TempStdDir::default();

        let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
            .storage_option("option1", "value1")
            .storage_option("option2", "value2")
            .build()
            .await
            .unwrap();

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table and check storage options are included
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        let storage_options = response.storage_options.unwrap();
        assert_eq!(storage_options.get("option1"), Some(&"value1".to_string()));
        assert_eq!(storage_options.get("option2"), Some(&"value2".to_string()));
    }

    /// When no credential vendor is configured, `describe_table` and
    /// `declare_table` must strip credential keys from storage options
    /// while preserving non-credential config (region, endpoint, etc.).
    #[tokio::test]
    async fn test_no_storage_options_without_vendor() {
        use lance_namespace::models::DeclareTableRequest;

        let temp_dir = TempStdDir::default();

        // No manifest, no credential vendor, but storage options with credentials
        let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
            .manifest_enabled(false)
            .storage_option("aws_access_key_id", "AKID")
            .storage_option("aws_secret_access_key", "SECRET")
            .storage_option("region", "us-east-1")
            .build()
            .await
            .unwrap();

        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // create_table
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["t1".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // describe_table should not return storage options without a vendor
        let mut desc_req = DescribeTableRequest::new();
        desc_req.id = Some(vec!["t1".to_string()]);
        let resp = namespace.describe_table(desc_req).await.unwrap();
        assert!(resp.storage_options.is_none());

        // declare_table should not return storage options without a vendor
        let mut decl_req = DeclareTableRequest::new();
        decl_req.id = Some(vec!["t2".to_string()]);
        let resp = namespace.declare_table(decl_req).await.unwrap();
        assert!(resp.storage_options.is_none());
    }

    /// Same test with manifest mode enabled.
    #[tokio::test]
    async fn test_no_storage_options_without_vendor_manifest() {
        let temp_dir = TempStdDir::default();

        let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
            .storage_option("aws_access_key_id", "AKID")
            .storage_option("aws_secret_access_key", "SECRET")
            .storage_option("region", "us-east-1")
            .build()
            .await
            .unwrap();

        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["t1".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // describe_table through manifest should not return storage options without a vendor
        let mut desc_req = DescribeTableRequest::new();
        desc_req.id = Some(vec!["t1".to_string()]);
        let resp = namespace.describe_table(desc_req).await.unwrap();
        assert!(resp.storage_options.is_none());
    }

    #[tokio::test]
    async fn test_from_properties_manifest_enabled() {
        let temp_dir = TempStdDir::default();

        let mut properties = HashMap::new();
        properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());
        properties.insert("manifest_enabled".to_string(), "true".to_string());
        properties.insert("dir_listing_enabled".to_string(), "false".to_string());

        let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
        assert!(builder.manifest_enabled);
        assert!(!builder.dir_listing_enabled);

        let namespace = builder.build().await.unwrap();

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.is_some());
    }

    #[tokio::test]
    async fn test_from_properties_dir_listing_enabled() {
        let temp_dir = TempStdDir::default();

        let mut properties = HashMap::new();
        properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());
        properties.insert("manifest_enabled".to_string(), "false".to_string());
        properties.insert("dir_listing_enabled".to_string(), "true".to_string());

        let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
        assert!(!builder.manifest_enabled);
        assert!(builder.dir_listing_enabled);

        let namespace = builder.build().await.unwrap();

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.is_some());
    }

    #[tokio::test]
    async fn test_from_properties_defaults() {
        let temp_dir = TempStdDir::default();

        let mut properties = HashMap::new();
        properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());

        let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
        // Both should default to true
        assert!(builder.manifest_enabled);
        assert!(builder.dir_listing_enabled);
    }

    #[tokio::test]
    async fn test_from_properties_with_storage_options() {
        let temp_dir = TempStdDir::default();

        let mut properties = HashMap::new();
        properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());
        properties.insert("manifest_enabled".to_string(), "true".to_string());
        properties.insert("storage.region".to_string(), "us-west-2".to_string());
        properties.insert("storage.bucket".to_string(), "my-bucket".to_string());

        let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
        assert!(builder.manifest_enabled);
        assert!(builder.storage_options.is_some());

        let storage_options = builder.storage_options.unwrap();
        assert_eq!(
            storage_options.get("region"),
            Some(&"us-west-2".to_string())
        );
        assert_eq!(
            storage_options.get("bucket"),
            Some(&"my-bucket".to_string())
        );
    }

    #[tokio::test]
    async fn test_various_arrow_types() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create schema with various types
        let fields = vec![
            JsonArrowField {
                name: "bool_col".to_string(),
                r#type: Box::new(JsonArrowDataType::new("bool".to_string())),
                nullable: true,
                metadata: None,
            },
            JsonArrowField {
                name: "int8_col".to_string(),
                r#type: Box::new(JsonArrowDataType::new("int8".to_string())),
                nullable: true,
                metadata: None,
            },
            JsonArrowField {
                name: "float64_col".to_string(),
                r#type: Box::new(JsonArrowDataType::new("float64".to_string())),
                nullable: true,
                metadata: None,
            },
            JsonArrowField {
                name: "binary_col".to_string(),
                r#type: Box::new(JsonArrowDataType::new("binary".to_string())),
                nullable: true,
                metadata: None,
            },
        ];

        let schema = JsonArrowSchema {
            fields,
            metadata: None,
        };

        // Create IPC data
        let ipc_data = create_test_ipc_data(&schema);

        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["complex_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.is_some());
    }

    #[tokio::test]
    async fn test_connect_dir() {
        let temp_dir = TempStdDir::default();

        let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
            .build()
            .await
            .unwrap();

        // Test basic operation through the concrete type
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);
    }

    #[tokio::test]
    async fn test_create_table_with_ipc_data() {
        use arrow::array::{Int32Array, StringArray};
        use arrow::ipc::writer::StreamWriter;

        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create a schema with some fields
        let schema = create_test_schema();

        // Create some test data that matches the schema
        let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
        let arrow_schema = Arc::new(arrow_schema);

        // Create a RecordBatch with actual data
        let id_array = Int32Array::from(vec![1, 2, 3]);
        let name_array = StringArray::from(vec!["Alice", "Bob", "Charlie"]);
        let batch = arrow::record_batch::RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(id_array), Arc::new(name_array)],
        )
        .unwrap();

        // Write the batch to an IPC stream
        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        // Create table with the IPC data
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table_with_data".to_string()]);

        let response = namespace
            .create_table(request, Bytes::from(buffer))
            .await
            .unwrap();

        assert_eq!(response.version, Some(1));
        assert!(
            response
                .location
                .unwrap()
                .contains("test_table_with_data.lance")
        );

        // Verify table exists
        let mut exists_request = TableExistsRequest::new();
        exists_request.id = Some(vec!["test_table_with_data".to_string()]);
        namespace.table_exists(exists_request).await.unwrap();
    }

    #[tokio::test]
    async fn test_child_namespace_create_and_list() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create multiple child namespaces
        for i in 1..=3 {
            let mut create_req = CreateNamespaceRequest::new();
            create_req.id = Some(vec![format!("ns{}", i)]);
            let result = namespace.create_namespace(create_req).await;
            assert!(result.is_ok(), "Failed to create child namespace ns{}", i);
        }

        // List child namespaces
        let list_req = ListNamespacesRequest {
            id: Some(vec![]),
            ..Default::default()
        };
        let result = namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let namespaces = result.unwrap().namespaces;
        assert_eq!(namespaces.len(), 3);
        assert!(namespaces.contains(&"ns1".to_string()));
        assert!(namespaces.contains(&"ns2".to_string()));
        assert!(namespaces.contains(&"ns3".to_string()));
    }

    #[tokio::test]
    async fn test_nested_namespace_hierarchy() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create parent namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        // Create nested children
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string(), "child1".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string(), "child2".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        // List children of parent
        let list_req = ListNamespacesRequest {
            id: Some(vec!["parent".to_string()]),
            ..Default::default()
        };
        let result = namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let children = result.unwrap().namespaces;
        assert_eq!(children.len(), 2);
        assert!(children.contains(&"child1".to_string()));
        assert!(children.contains(&"child2".to_string()));

        // List root should only show parent
        let list_req = ListNamespacesRequest {
            id: Some(vec![]),
            ..Default::default()
        };
        let result = namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let root_namespaces = result.unwrap().namespaces;
        assert_eq!(root_namespaces.len(), 1);
        assert_eq!(root_namespaces[0], "parent");
    }

    #[tokio::test]
    async fn test_table_in_child_namespace() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create table in child namespace
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await;
        assert!(result.is_ok(), "Failed to create table in child namespace");

        // List tables in child namespace
        let list_req = ListTablesRequest {
            id: Some(vec!["test_ns".to_string()]),
            ..Default::default()
        };
        let result = namespace.list_tables(list_req).await;
        assert!(result.is_ok());
        let tables = result.unwrap().tables;
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0], "table1");

        // Verify table exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_ok());

        // Describe table in child namespace
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace.describe_table(describe_req).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.location.is_some());
    }

    #[tokio::test]
    async fn test_multiple_tables_in_child_namespace() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create multiple tables
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        for i in 1..=3 {
            let mut create_table_req = CreateTableRequest::new();
            create_table_req.id = Some(vec!["test_ns".to_string(), format!("table{}", i)]);
            namespace
                .create_table(create_table_req, bytes::Bytes::from(ipc_data.clone()))
                .await
                .unwrap();
        }

        // List tables
        let list_req = ListTablesRequest {
            id: Some(vec!["test_ns".to_string()]),
            ..Default::default()
        };
        let result = namespace.list_tables(list_req).await;
        assert!(result.is_ok());
        let tables = result.unwrap().tables;
        assert_eq!(tables.len(), 3);
        assert!(tables.contains(&"table1".to_string()));
        assert!(tables.contains(&"table2".to_string()));
        assert!(tables.contains(&"table3".to_string()));
    }

    #[tokio::test]
    async fn test_drop_table_in_child_namespace() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Drop table
        let mut drop_req = DropTableRequest::new();
        drop_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace.drop_table(drop_req).await;
        assert!(result.is_ok(), "Failed to drop table in child namespace");

        // Verify table no longer exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_deeply_nested_namespace() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create deeply nested namespace hierarchy
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["level1".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["level1".to_string(), "level2".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec![
            "level1".to_string(),
            "level2".to_string(),
            "level3".to_string(),
        ]);
        namespace.create_namespace(create_req).await.unwrap();

        // Create table in deeply nested namespace
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec![
            "level1".to_string(),
            "level2".to_string(),
            "level3".to_string(),
            "table1".to_string(),
        ]);
        let result = namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await;
        assert!(
            result.is_ok(),
            "Failed to create table in deeply nested namespace"
        );

        // Verify table exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec![
            "level1".to_string(),
            "level2".to_string(),
            "level3".to_string(),
            "table1".to_string(),
        ]);
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_namespace_with_properties() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create namespace with properties
        let mut properties = HashMap::new();
        properties.insert("owner".to_string(), "test_user".to_string());
        properties.insert("description".to_string(), "Test namespace".to_string());

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["test_ns".to_string()]);
        create_req.properties = Some(properties.clone());
        namespace.create_namespace(create_req).await.unwrap();

        // Describe namespace and verify properties
        let describe_req = DescribeNamespaceRequest {
            id: Some(vec!["test_ns".to_string()]),
            ..Default::default()
        };
        let result = namespace.describe_namespace(describe_req).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.properties.is_some());
        let props = response.properties.unwrap();
        assert_eq!(props.get("owner"), Some(&"test_user".to_string()));
        assert_eq!(
            props.get("description"),
            Some(&"Test namespace".to_string())
        );
    }

    #[tokio::test]
    async fn test_cannot_drop_namespace_with_tables() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create table in namespace
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Try to drop namespace - should fail
        let mut drop_req = DropNamespaceRequest::new();
        drop_req.id = Some(vec!["test_ns".to_string()]);
        let result = namespace.drop_namespace(drop_req).await;
        assert!(
            result.is_err(),
            "Should not be able to drop namespace with tables"
        );
    }

    #[tokio::test]
    async fn test_isolation_between_namespaces() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create two namespaces
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns1".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns2".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        // Create table with same name in both namespaces
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
        namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data.clone()))
            .await
            .unwrap();

        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["ns2".to_string(), "table1".to_string()]);
        namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // List tables in each namespace
        let list_req = ListTablesRequest {
            id: Some(vec!["ns1".to_string()]),
            page_token: None,
            limit: None,
            ..Default::default()
        };
        let result = namespace.list_tables(list_req).await.unwrap();
        assert_eq!(result.tables.len(), 1);
        assert_eq!(result.tables[0], "table1");

        let list_req = ListTablesRequest {
            id: Some(vec!["ns2".to_string()]),
            page_token: None,
            limit: None,
            ..Default::default()
        };
        let result = namespace.list_tables(list_req).await.unwrap();
        assert_eq!(result.tables.len(), 1);
        assert_eq!(result.tables[0], "table1");

        // Drop table in ns1 shouldn't affect ns2
        let mut drop_req = DropTableRequest::new();
        drop_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
        namespace.drop_table(drop_req).await.unwrap();

        // Verify ns1 table is gone but ns2 table still exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_err());

        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["ns2".to_string(), "table1".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_ok());
    }

    #[tokio::test]
    async fn test_migrate_directory_tables() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Step 1: Create tables in directory-only mode
        let dir_only_ns = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create some tables
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        for i in 1..=3 {
            let mut create_req = CreateTableRequest::new();
            create_req.id = Some(vec![format!("table{}", i)]);
            dir_only_ns
                .create_table(create_req, bytes::Bytes::from(ipc_data.clone()))
                .await
                .unwrap();
        }

        drop(dir_only_ns);

        // Step 2: Create namespace with dual mode (manifest + directory listing)
        let dual_mode_ns = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Before migration, tables should be visible (via directory listing fallback)
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let tables = dual_mode_ns.list_tables(list_req).await.unwrap().tables;
        assert_eq!(tables.len(), 3);

        // Run migration
        let migrated_count = dual_mode_ns.migrate().await.unwrap();
        assert_eq!(migrated_count, 3, "Should migrate all 3 tables");

        // Verify tables are now in manifest
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let tables = dual_mode_ns.list_tables(list_req).await.unwrap().tables;
        assert_eq!(tables.len(), 3);

        // Run migration again - should be idempotent
        let migrated_count = dual_mode_ns.migrate().await.unwrap();
        assert_eq!(
            migrated_count, 0,
            "Should not migrate already-migrated tables"
        );

        drop(dual_mode_ns);

        // Step 3: Create namespace with manifest-only mode
        let manifest_only_ns = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        // Tables should still be accessible (now from manifest only)
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let tables = manifest_only_ns.list_tables(list_req).await.unwrap().tables;
        assert_eq!(tables.len(), 3);
        assert!(tables.contains(&"table1".to_string()));
        assert!(tables.contains(&"table2".to_string()));
        assert!(tables.contains(&"table3".to_string()));
    }

    #[tokio::test]
    async fn test_migrate_without_manifest() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace without manifest
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // migrate() should return 0 when manifest is not enabled
        let migrated_count = namespace.migrate().await.unwrap();
        assert_eq!(migrated_count, 0);
    }

    #[tokio::test]
    async fn test_register_table() {
        use lance_namespace::models::{RegisterTableRequest, TableExistsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .dir_listing_to_manifest_migration_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a physical table first using lance directly
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let table_uri = format!("{}/external_table.lance", temp_path);
        let cursor = Cursor::new(ipc_data);
        let stream_reader = StreamReader::try_new(cursor, None).unwrap();
        let batches: Vec<_> = stream_reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let schema = batches[0].schema();
        let batch_results: Vec<_> = batches.into_iter().map(Ok).collect();
        let reader = RecordBatchIterator::new(batch_results, schema);
        Dataset::write(Box::new(reader), &table_uri, None)
            .await
            .unwrap();

        // Register the table
        let mut register_req = RegisterTableRequest::new("external_table.lance".to_string());
        register_req.id = Some(vec!["registered_table".to_string()]);

        let response = namespace.register_table(register_req).await.unwrap();
        assert_eq!(response.location, Some("external_table.lance".to_string()));

        // Verify table exists in namespace
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["registered_table".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_ok());

        // Verify we can list the table
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let tables = namespace.list_tables(list_req).await.unwrap();
        assert!(tables.tables.contains(&"registered_table".to_string()));
    }

    #[tokio::test]
    async fn test_register_table_duplicate_fails() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Register a table
        let mut register_req = RegisterTableRequest::new("test_table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);

        namespace
            .register_table(register_req.clone())
            .await
            .unwrap();

        // Try to register again - should fail
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn test_deregister_table() {
        use lance_namespace::models::{DeregisterTableRequest, TableExistsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace with manifest-only mode (no directory listing fallback)
        // This ensures deregistered tables are truly invisible
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Verify table exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req.clone()).await.is_ok());

        // Deregister the table
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        let response = namespace.deregister_table(deregister_req).await.unwrap();

        // Should return location and id
        assert!(
            response.location.is_some(),
            "Deregister should return location"
        );
        let location = response.location.as_ref().unwrap();
        // Location should be a proper file:// URI with the temp path
        // Use uri_to_url to normalize the temp path to a URL for comparison
        let expected_url = lance_io::object_store::uri_to_url(temp_path)
            .expect("Failed to convert temp path to URL");
        let expected_prefix = expected_url.to_string();
        assert!(
            location.starts_with(&expected_prefix),
            "Location should start with '{}', got: {}",
            expected_prefix,
            location
        );
        assert!(
            location.contains("test_table"),
            "Location should contain table name: {}",
            location
        );
        assert_eq!(response.id, Some(vec!["test_table".to_string()]));

        // Verify table no longer exists in namespace (removed from manifest)
        assert!(namespace.table_exists(exists_req).await.is_err());

        // Verify physical data still exists at the returned location
        let dataset = Dataset::open(location).await;
        assert!(
            dataset.is_ok(),
            "Physical table data should still exist at {}",
            location
        );
    }

    #[tokio::test]
    async fn test_deregister_table_in_child_namespace() {
        use lance_namespace::models::{
            CreateNamespaceRequest, DeregisterTableRequest, TableExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Create child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create a table in the child namespace
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_ns".to_string(), "test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Deregister the table
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_ns".to_string(), "test_table".to_string()]);
        let response = namespace.deregister_table(deregister_req).await.unwrap();

        // Should return location and id in child namespace
        assert!(
            response.location.is_some(),
            "Deregister should return location"
        );
        let location = response.location.as_ref().unwrap();
        // Location should be a proper file:// URI with the temp path
        // Use uri_to_url to normalize the temp path to a URL for comparison
        let expected_url = lance_io::object_store::uri_to_url(temp_path)
            .expect("Failed to convert temp path to URL");
        let expected_prefix = expected_url.to_string();
        assert!(
            location.starts_with(&expected_prefix),
            "Location should start with '{}', got: {}",
            expected_prefix,
            location
        );
        assert!(
            location.contains("test_ns") && location.contains("test_table"),
            "Location should contain namespace and table name: {}",
            location
        );
        assert_eq!(
            response.id,
            Some(vec!["test_ns".to_string(), "test_table".to_string()])
        );

        // Verify table no longer exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_ns".to_string(), "test_table".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_err());
    }

    #[tokio::test]
    async fn test_register_without_manifest_fails() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace without manifest
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();

        // Try to register - should fail (register requires manifest)
        let mut register_req = RegisterTableRequest::new("test_table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("manifest mode is enabled")
        );

        // Note: deregister_table now works in V1 mode via .lance-deregistered marker files
        // See test_deregister_table_v1_mode for that test case
    }

    #[tokio::test]
    async fn test_register_table_rejects_absolute_uri() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Try to register with absolute URI - should fail
        let mut register_req = RegisterTableRequest::new("s3://bucket/table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Absolute URIs are not allowed"));
    }

    #[tokio::test]
    async fn test_register_table_rejects_absolute_path() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Try to register with absolute path - should fail
        let mut register_req = RegisterTableRequest::new("/tmp/table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Absolute paths are not allowed"));
    }

    #[tokio::test]
    async fn test_register_table_rejects_path_traversal() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Try to register with path traversal - should fail
        let mut register_req = RegisterTableRequest::new("../outside/table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Path traversal is not allowed"));
    }

    #[tokio::test]
    async fn test_namespace_write() {
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use arrow::record_batch::{RecordBatch, RecordBatchIterator};
        use lance::dataset::{Dataset, WriteMode, WriteParams};
        use lance_namespace::LanceNamespace;

        let (namespace, _temp_dir) = create_test_namespace().await;
        let namespace = Arc::new(namespace) as Arc<dyn LanceNamespace>;

        // Use child namespace instead of root
        let table_id = vec!["test_ns".to_string(), "test_table".to_string()];
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, false),
        ]));

        // Test 1: CREATE mode
        let data1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();

        let reader1 = RecordBatchIterator::new(vec![data1].into_iter().map(Ok), schema.clone());
        let dataset =
            Dataset::write_into_namespace(reader1, namespace.clone(), table_id.clone(), None)
                .await
                .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
        assert_eq!(dataset.version().version, 1);

        // Test 2: APPEND mode
        let data2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![4, 5])),
                Arc::new(Int32Array::from(vec![40, 50])),
            ],
        )
        .unwrap();

        let params_append = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };

        let reader2 = RecordBatchIterator::new(vec![data2].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write_into_namespace(
            reader2,
            namespace.clone(),
            table_id.clone(),
            Some(params_append),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);
        assert_eq!(dataset.version().version, 2);

        // Test 3: OVERWRITE mode
        let data3 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![100, 200])),
                Arc::new(Int32Array::from(vec![1000, 2000])),
            ],
        )
        .unwrap();

        let params_overwrite = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };

        let reader3 = RecordBatchIterator::new(vec![data3].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write_into_namespace(
            reader3,
            namespace.clone(),
            table_id.clone(),
            Some(params_overwrite),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
        assert_eq!(dataset.version().version, 3);

        // Verify old data was replaced
        let result = dataset.scan().try_into_batch().await.unwrap();
        let a_col = result
            .column_by_name("a")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(a_col.values(), &[100, 200]);
    }

    // ============================================================
    // Tests for declare_table
    // ============================================================

    #[tokio::test]
    async fn test_declare_table_v1_mode() {
        use lance_namespace::models::{
            DeclareTableRequest, DescribeTableRequest, ListTablesRequest, TableExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace in V1 mode (no manifest)
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();

        // Declare a table
        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        let response = namespace.declare_table(declare_req).await.unwrap();

        // Should return location
        assert!(response.location.is_some());
        let location = response.location.as_ref().unwrap();
        assert!(location.ends_with("test_table.lance"));

        // Table should exist (via reserved file)
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_ok());

        // Describe should work but return no version/schema (not written yet)
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert!(describe_response.location.is_some());
        assert!(describe_response.version.is_none()); // Not written yet
        assert!(describe_response.schema.is_none()); // Not written yet
        assert_eq!(describe_response.is_only_declared, None);

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        describe_req.check_declared = Some(true);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(describe_response.is_only_declared, Some(true));

        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let list_response = namespace.list_tables(list_req.clone()).await.unwrap();
        assert_eq!(list_response.tables, vec!["test_table".to_string()]);

        list_req.include_declared = Some(false);
        let list_response = namespace.list_tables(list_req).await.unwrap();
        assert!(list_response.tables.is_empty());
    }

    #[tokio::test]
    async fn test_insert_into_declared_table_promotes_it_from_declared_state() {
        use lance_namespace::models::{
            DeclareTableRequest, DescribeTableRequest, InsertIntoTableRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();

        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        namespace.declare_table(declare_req).await.unwrap();

        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut insert_req = InsertIntoTableRequest::new();
        insert_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .insert_into_table(insert_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        describe_req.load_detailed_metadata = Some(true);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();

        assert_eq!(describe_response.is_only_declared, Some(false));
        assert_eq!(describe_response.version, Some(1));
        assert!(describe_response.schema.is_some());

        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        list_req.include_declared = Some(false);
        assert_eq!(
            namespace.list_tables(list_req).await.unwrap().tables,
            vec!["test_table".to_string()]
        );
    }

    #[tokio::test]
    async fn test_create_table_after_declare_table_v1_mode_creates_table() {
        use lance_namespace::models::{
            DeclareTableRequest, DescribeTableRequest, ListTablesRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();

        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        namespace.declare_table(declare_req).await.unwrap();

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        let response = namespace
            .create_table(
                create_req,
                bytes::Bytes::from(create_non_empty_test_ipc_data()),
            )
            .await
            .unwrap();

        assert_eq!(response.version, Some(1));

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        describe_req.load_detailed_metadata = Some(true);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(describe_response.is_only_declared, Some(false));
        assert_eq!(describe_response.version, Some(1));

        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        list_req.include_declared = Some(false);
        assert_eq!(
            namespace.list_tables(list_req).await.unwrap().tables,
            vec!["test_table".to_string()]
        );
    }

    #[tokio::test]
    async fn test_insert_into_declared_table_with_manifest_promotes_it() {
        use lance_namespace::models::{
            DeclareTableRequest, DescribeTableRequest, InsertIntoTableRequest, ListTablesRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        namespace.declare_table(declare_req).await.unwrap();

        let mut insert_req = InsertIntoTableRequest::new();
        insert_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .insert_into_table(
                insert_req,
                bytes::Bytes::from(create_non_empty_test_ipc_data()),
            )
            .await
            .unwrap();

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        describe_req.load_detailed_metadata = Some(true);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(describe_response.is_only_declared, Some(false));
        assert_eq!(describe_response.version, Some(1));

        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        list_req.include_declared = Some(false);
        assert_eq!(
            namespace.list_tables(list_req).await.unwrap().tables,
            vec!["test_table".to_string()]
        );
    }

    #[tokio::test]
    async fn test_create_table_after_declare_table_with_manifest_creates_table() {
        use lance_namespace::models::{
            CreateTableRequest, DeclareTableRequest, DescribeTableRequest, ListTablesRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        declare_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
        namespace.declare_table(declare_req).await.unwrap();

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        create_req.mode = Some("Overwrite".to_string());
        let response = namespace
            .create_table(
                create_req,
                bytes::Bytes::from(create_non_empty_test_ipc_data()),
            )
            .await
            .unwrap();

        assert_eq!(response.version, Some(1));
        assert_eq!(
            response
                .properties
                .as_ref()
                .and_then(|properties| properties.get("owner")),
            Some(&"alice".to_string())
        );

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        describe_req.load_detailed_metadata = Some(true);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(describe_response.is_only_declared, Some(false));
        assert_eq!(describe_response.version, Some(1));
        assert_eq!(
            describe_response
                .properties
                .as_ref()
                .and_then(|properties| properties.get("owner")),
            Some(&"alice".to_string())
        );

        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        list_req.include_declared = Some(false);
        assert_eq!(
            namespace.list_tables(list_req).await.unwrap().tables,
            vec!["test_table".to_string()]
        );
    }

    #[tokio::test]
    async fn test_create_table_after_declare_table_with_manifest_rejects_new_properties() {
        use lance_namespace::models::{CreateTableRequest, DeclareTableRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        declare_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
        namespace.declare_table(declare_req).await.unwrap();

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        create_req.properties = Some(HashMap::from([("owner".to_string(), "bob".to_string())]));

        let result = namespace
            .create_table(
                create_req,
                bytes::Bytes::from(create_non_empty_test_ipc_data()),
            )
            .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("cannot set properties for already declared table")
        );
    }

    #[tokio::test]
    async fn test_create_table_with_manifest_exist_ok_keeps_existing_table() {
        use lance_namespace::models::{CreateTableRequest, DescribeTableRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        create_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
        namespace
            .create_table(
                create_req,
                bytes::Bytes::from(create_non_empty_test_ipc_data()),
            )
            .await
            .unwrap();

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        create_req.mode = Some("ExistOk".to_string());
        create_req.properties = Some(HashMap::from([("owner".to_string(), "bob".to_string())]));
        let response = namespace
            .create_table(
                create_req,
                bytes::Bytes::from(create_single_row_test_ipc_data()),
            )
            .await
            .unwrap();

        assert_eq!(
            response
                .properties
                .as_ref()
                .and_then(|properties| properties.get("owner")),
            Some(&"alice".to_string())
        );
        assert_eq!(
            open_dataset(&namespace, "test_table")
                .await
                .count_rows(None)
                .await
                .unwrap(),
            2
        );

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(
            describe_response
                .properties
                .as_ref()
                .and_then(|properties| properties.get("owner")),
            Some(&"alice".to_string())
        );
    }

    #[tokio::test]
    async fn test_create_table_with_manifest_overwrite_replaces_existing_table() {
        use lance_namespace::models::{CreateTableRequest, DescribeTableRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        create_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
        namespace
            .create_table(
                create_req,
                bytes::Bytes::from(create_non_empty_test_ipc_data()),
            )
            .await
            .unwrap();

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        create_req.mode = Some("overwrite".to_string());
        create_req.properties = Some(HashMap::from([("owner".to_string(), "bob".to_string())]));
        let response = namespace
            .create_table(
                create_req,
                bytes::Bytes::from(create_single_row_test_ipc_data()),
            )
            .await
            .unwrap();

        assert_eq!(response.version, Some(2));
        assert_eq!(
            response
                .properties
                .as_ref()
                .and_then(|properties| properties.get("owner")),
            Some(&"bob".to_string())
        );
        assert_eq!(
            open_dataset(&namespace, "test_table")
                .await
                .count_rows(None)
                .await
                .unwrap(),
            1
        );

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(
            describe_response
                .properties
                .as_ref()
                .and_then(|properties| properties.get("owner")),
            Some(&"bob".to_string())
        );
    }

    #[tokio::test]
    async fn test_create_table_with_manifest_invalid_mode_rejected() {
        use lance_namespace::models::CreateTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        create_req.mode = Some("append".to_string());
        let result = namespace
            .create_table(
                create_req,
                bytes::Bytes::from(create_non_empty_test_ipc_data()),
            )
            .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported create_table mode")
        );
    }

    #[tokio::test]
    async fn test_merge_insert_into_declared_table_v1_mode_creates_table() {
        use lance_namespace::models::{
            DeclareTableRequest, DescribeTableRequest, ListTablesRequest,
            MergeInsertIntoTableRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();

        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        namespace.declare_table(declare_req).await.unwrap();

        let mut merge_req = MergeInsertIntoTableRequest::new();
        merge_req.id = Some(vec!["test_table".to_string()]);
        merge_req.on = Some("id".to_string());
        let response = namespace
            .merge_insert_into_table(
                merge_req,
                bytes::Bytes::from(create_non_empty_test_ipc_data()),
            )
            .await
            .unwrap();

        assert_eq!(response.num_inserted_rows, Some(2));
        assert_eq!(response.num_updated_rows, Some(0));

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        describe_req.load_detailed_metadata = Some(true);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(describe_response.is_only_declared, Some(false));
        assert_eq!(describe_response.version, Some(1));

        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        list_req.include_declared = Some(false);
        assert_eq!(
            namespace.list_tables(list_req).await.unwrap().tables,
            vec!["test_table".to_string()]
        );
    }

    #[tokio::test]
    async fn test_merge_insert_into_declared_table_with_manifest_creates_table() {
        use lance_namespace::models::{
            DeclareTableRequest, DescribeTableRequest, ListTablesRequest,
            MergeInsertIntoTableRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        namespace.declare_table(declare_req).await.unwrap();

        let mut merge_req = MergeInsertIntoTableRequest::new();
        merge_req.id = Some(vec!["test_table".to_string()]);
        merge_req.on = Some("id".to_string());
        let response = namespace
            .merge_insert_into_table(
                merge_req,
                bytes::Bytes::from(create_non_empty_test_ipc_data()),
            )
            .await
            .unwrap();

        assert_eq!(response.num_inserted_rows, Some(2));
        assert_eq!(response.num_updated_rows, Some(0));

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        describe_req.load_detailed_metadata = Some(true);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(describe_response.is_only_declared, Some(false));
        assert_eq!(describe_response.version, Some(1));

        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        list_req.include_declared = Some(false);
        assert_eq!(
            namespace.list_tables(list_req).await.unwrap().tables,
            vec!["test_table".to_string()]
        );
    }

    #[tokio::test]
    async fn test_declare_table_with_manifest() {
        use lance_namespace::models::{
            DeclareTableRequest, DescribeTableRequest, ListTablesRequest, TableExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace with manifest
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        // Declare a table
        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        declare_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
        let response = namespace.declare_table(declare_req).await.unwrap();

        // Should return location
        assert!(response.location.is_some());
        assert_eq!(
            response
                .properties
                .as_ref()
                .and_then(|properties| properties.get("owner")),
            Some(&"alice".to_string())
        );

        // Table should exist in manifest
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_ok());

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(describe_response.is_only_declared, None);

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        describe_req.check_declared = Some(true);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert_eq!(describe_response.is_only_declared, Some(true));
        assert_eq!(
            describe_response
                .properties
                .as_ref()
                .and_then(|properties| properties.get("owner")),
            Some(&"alice".to_string())
        );

        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        assert_eq!(
            namespace
                .list_tables(list_req.clone())
                .await
                .unwrap()
                .tables,
            vec!["test_table".to_string()]
        );
        list_req.include_declared = Some(false);
        assert!(
            namespace
                .list_tables(list_req)
                .await
                .unwrap()
                .tables
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_declare_table_when_table_exists() {
        use lance_namespace::models::DeclareTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();

        // First create a table with actual data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Try to declare the same table - should fail because it already has data
        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.declare_table(declare_req).await;
        assert!(result.is_err());
    }

    // ============================================================
    // Tests for deregister_table in V1 mode
    // ============================================================

    #[tokio::test]
    async fn test_deregister_table_v1_mode() {
        use lance_namespace::models::{DeregisterTableRequest, TableExistsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace in V1 mode (no manifest, with dir listing)
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table with data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Verify table exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req.clone()).await.is_ok());

        // Deregister the table
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        let response = namespace.deregister_table(deregister_req).await.unwrap();

        // Should return location
        assert!(response.location.is_some());
        let location = response.location.as_ref().unwrap();
        assert!(location.contains("test_table"));

        // Table should no longer exist (deregistered)
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("deregistered"));

        // Physical data should still exist
        let dataset = Dataset::open(location).await;
        assert!(dataset.is_ok(), "Physical table data should still exist");
    }

    #[tokio::test]
    async fn test_deregister_table_v1_already_deregistered() {
        use lance_namespace::models::DeregisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Deregister once
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .deregister_table(deregister_req.clone())
            .await
            .unwrap();

        // Try to deregister again - should fail
        let result = namespace.deregister_table(deregister_req).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("already deregistered")
        );
    }

    // ============================================================
    // Tests for list_tables skipping deregistered tables
    // ============================================================

    #[tokio::test]
    async fn test_list_tables_skips_deregistered_v1() {
        use lance_namespace::models::DeregisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create two tables
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_req1 = CreateTableRequest::new();
        create_req1.id = Some(vec!["table1".to_string()]);
        namespace
            .create_table(create_req1, bytes::Bytes::from(ipc_data.clone()))
            .await
            .unwrap();

        let mut create_req2 = CreateTableRequest::new();
        create_req2.id = Some(vec!["table2".to_string()]);
        namespace
            .create_table(create_req2, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // List tables - should see both (root namespace = empty vec)
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let list_response = namespace.list_tables(list_req.clone()).await.unwrap();
        assert_eq!(list_response.tables.len(), 2);

        // Deregister table1
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["table1".to_string()]);
        namespace.deregister_table(deregister_req).await.unwrap();

        // List tables - should only see table2
        let list_response = namespace.list_tables(list_req).await.unwrap();
        assert_eq!(list_response.tables.len(), 1);
        assert!(list_response.tables.contains(&"table2".to_string()));
        assert!(!list_response.tables.contains(&"table1".to_string()));
    }

    // ============================================================
    // Tests for describe_table and table_exists with deregistered tables
    // ============================================================

    #[tokio::test]
    async fn test_describe_table_fails_for_deregistered_v1() {
        use lance_namespace::models::{DeregisterTableRequest, DescribeTableRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Describe should work before deregistration
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.describe_table(describe_req.clone()).await.is_ok());

        // Deregister
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        namespace.deregister_table(deregister_req).await.unwrap();

        // Describe should fail after deregistration
        let result = namespace.describe_table(describe_req).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::Namespace { .. }));
        let err_msg = err.to_string();
        assert!(err_msg.contains("deregistered"));
        assert!(err_msg.contains("table id 'test_table'"));
    }

    #[tokio::test]
    async fn test_table_exists_fails_for_deregistered_v1() {
        use lance_namespace::models::{DeregisterTableRequest, TableExistsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Table exists should work before deregistration
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req.clone()).await.is_ok());

        // Deregister
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        namespace.deregister_table(deregister_req).await.unwrap();

        // Table exists should fail after deregistration
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::Namespace { .. }));
        let err_msg = err.to_string();
        assert!(err_msg.contains("deregistered"));
        assert!(err_msg.contains("table id 'test_table'"));
    }

    #[tokio::test]
    async fn test_atomic_table_status_check() {
        // This test verifies that the TableStatus check is atomic
        // by ensuring a single directory listing is used

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Table status should show exists=true, is_deregistered=false
        let status = namespace.check_table_status("test_table").await;
        assert!(status.exists);
        assert!(!status.is_deregistered);
        assert!(!status.has_reserved_file);
    }

    #[tokio::test]
    async fn test_table_version_tracking_enabled_managed_versioning() {
        use lance_namespace::models::DescribeTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace with table_version_tracking_enabled=true
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Describe table should return managed_versioning=true
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        let describe_resp = namespace.describe_table(describe_req).await.unwrap();

        // managed_versioning should be true
        assert_eq!(
            describe_resp.managed_versioning,
            Some(true),
            "managed_versioning should be true when table_version_tracking_enabled=true"
        );
    }

    #[tokio::test]
    async fn test_table_version_tracking_disabled_no_managed_versioning() {
        use lance_namespace::models::DescribeTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace with table_version_tracking_enabled=false (default)
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(false)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Describe table should not have managed_versioning set
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        let describe_resp = namespace.describe_table(describe_req).await.unwrap();

        // managed_versioning should be None when table_version_tracking_enabled=false
        assert!(
            describe_resp.managed_versioning.is_none(),
            "managed_versioning should be None when table_version_tracking_enabled=false, got: {:?}",
            describe_resp.managed_versioning
        );
    }

    #[tokio::test]
    async fn test_list_table_versions() {
        use arrow::array::{Int32Array, RecordBatchIterator};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use lance::dataset::{Dataset, WriteMode, WriteParams};
        use lance_namespace::models::{CreateNamespaceRequest, ListTableVersionsRequest};

        let temp_dir = TempStrDir::default();
        let temp_path: &str = &temp_dir;

        let namespace: Arc<dyn LanceNamespace> = Arc::new(
            DirectoryNamespaceBuilder::new(temp_path)
                .table_version_tracking_enabled(true)
                .build()
                .await
                .unwrap(),
        );

        // Create parent namespace first
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["workspace".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create a table using write_into_namespace (version 1)
        let table_id = vec!["workspace".to_string(), "test_table".to_string()];
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch.clone())], arrow_schema.clone());
        let write_params = WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        };
        let mut dataset = Dataset::write_into_namespace(
            batches,
            namespace.clone(),
            table_id.clone(),
            Some(write_params),
        )
        .await
        .unwrap();

        // Append to create version 2
        let batch2 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![100, 200]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch2)], arrow_schema.clone());
        dataset.append(batches, None).await.unwrap();

        // Append to create version 3
        let batch3 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![300, 400]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch3)], arrow_schema);
        dataset.append(batches, None).await.unwrap();

        // List versions - should have versions 1, 2, and 3
        let mut list_req = ListTableVersionsRequest::new();
        list_req.id = Some(table_id.clone());
        let list_resp = namespace.list_table_versions(list_req).await.unwrap();

        assert_eq!(
            list_resp.versions.len(),
            3,
            "Should have 3 versions, got: {:?}",
            list_resp.versions
        );

        // Verify each version
        for expected_version in 1..=3 {
            let version = list_resp
                .versions
                .iter()
                .find(|v| v.version == expected_version)
                .unwrap_or_else(|| panic!("Expected version {}", expected_version));

            assert!(
                !version.manifest_path.is_empty(),
                "manifest_path should be set for version {}",
                expected_version
            );
            assert!(
                version.manifest_path.contains(".manifest"),
                "manifest_path should contain .manifest for version {}",
                expected_version
            );
            assert!(
                version.manifest_size.is_some(),
                "manifest_size should be set for version {}",
                expected_version
            );
            assert!(
                version.manifest_size.unwrap() > 0,
                "manifest_size should be > 0 for version {}",
                expected_version
            );
            assert!(
                version.timestamp_millis.is_some(),
                "timestamp_millis should be set for version {}",
                expected_version
            );
        }
    }

    #[tokio::test]
    async fn test_describe_table_version() {
        use arrow::array::{Int32Array, RecordBatchIterator};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use lance::dataset::{Dataset, WriteMode, WriteParams};
        use lance_namespace::models::{CreateNamespaceRequest, DescribeTableVersionRequest};

        let temp_dir = TempStrDir::default();
        let temp_path: &str = &temp_dir;

        let namespace: Arc<dyn LanceNamespace> = Arc::new(
            DirectoryNamespaceBuilder::new(temp_path)
                .table_version_tracking_enabled(true)
                .build()
                .await
                .unwrap(),
        );

        // Create parent namespace first
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["workspace".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create a table using write_into_namespace (version 1)
        let table_id = vec!["workspace".to_string(), "test_table".to_string()];
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema.clone());
        let write_params = WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        };
        let mut dataset = Dataset::write_into_namespace(
            batches,
            namespace.clone(),
            table_id.clone(),
            Some(write_params),
        )
        .await
        .unwrap();

        // Append data to create version 2
        let batch2 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![100, 200]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch2)], arrow_schema);
        dataset.append(batches, None).await.unwrap();

        // Describe version 1
        let mut describe_req = DescribeTableVersionRequest::new();
        describe_req.id = Some(table_id.clone());
        describe_req.version = Some(1);
        let describe_resp = namespace
            .describe_table_version(describe_req)
            .await
            .unwrap();

        let version = &describe_resp.version;
        assert_eq!(version.version, 1);
        assert!(version.timestamp_millis.is_some());
        assert!(
            !version.manifest_path.is_empty(),
            "manifest_path should be set"
        );
        assert!(
            version.manifest_path.contains(".manifest"),
            "manifest_path should contain .manifest"
        );
        assert!(
            version.manifest_size.is_some(),
            "manifest_size should be set"
        );
        assert!(
            version.manifest_size.unwrap() > 0,
            "manifest_size should be > 0"
        );

        // Describe version 2
        let mut describe_req = DescribeTableVersionRequest::new();
        describe_req.id = Some(table_id.clone());
        describe_req.version = Some(2);
        let describe_resp = namespace
            .describe_table_version(describe_req)
            .await
            .unwrap();

        let version = &describe_resp.version;
        assert_eq!(version.version, 2);
        assert!(version.timestamp_millis.is_some());
        assert!(
            !version.manifest_path.is_empty(),
            "manifest_path should be set"
        );
        assert!(
            version.manifest_size.is_some(),
            "manifest_size should be set"
        );
        assert!(
            version.manifest_size.unwrap() > 0,
            "manifest_size should be > 0"
        );
    }

    #[tokio::test]
    async fn test_describe_table_version_latest() {
        use arrow::array::{Int32Array, RecordBatchIterator};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use lance::dataset::{Dataset, WriteMode, WriteParams};
        use lance_namespace::models::{CreateNamespaceRequest, DescribeTableVersionRequest};

        let temp_dir = TempStrDir::default();
        let temp_path: &str = &temp_dir;

        let namespace: Arc<dyn LanceNamespace> = Arc::new(
            DirectoryNamespaceBuilder::new(temp_path)
                .table_version_tracking_enabled(true)
                .build()
                .await
                .unwrap(),
        );

        // Create parent namespace first
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["workspace".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create a table using write_into_namespace (version 1)
        let table_id = vec!["workspace".to_string(), "test_table".to_string()];
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema.clone());
        let write_params = WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        };
        let mut dataset = Dataset::write_into_namespace(
            batches,
            namespace.clone(),
            table_id.clone(),
            Some(write_params),
        )
        .await
        .unwrap();

        // Append to create version 2
        let batch2 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![100, 200]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch2)], arrow_schema.clone());
        dataset.append(batches, None).await.unwrap();

        // Append to create version 3
        let batch3 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![300, 400]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch3)], arrow_schema);
        dataset.append(batches, None).await.unwrap();

        // Describe latest version (no version specified)
        let mut describe_req = DescribeTableVersionRequest::new();
        describe_req.id = Some(table_id.clone());
        describe_req.version = None;
        let describe_resp = namespace
            .describe_table_version(describe_req)
            .await
            .unwrap();

        // Should return version 3 as it's the latest
        assert_eq!(describe_resp.version.version, 3);
    }

    #[tokio::test]
    async fn test_create_table_version() {
        use futures::TryStreamExt;
        use lance::dataset::builder::DatasetBuilder;
        use lance_namespace::models::CreateTableVersionRequest;

        let temp_dir = TempStrDir::default();
        let temp_path: &str = &temp_dir;

        let namespace: Arc<dyn LanceNamespace> = Arc::new(
            DirectoryNamespaceBuilder::new(temp_path)
                .table_version_tracking_enabled(true)
                .build()
                .await
                .unwrap(),
        );

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Open the dataset using from_namespace to get proper object_store and paths
        let table_id = vec!["test_table".to_string()];
        let dataset = DatasetBuilder::from_namespace(namespace.clone(), table_id.clone())
            .await
            .unwrap()
            .load()
            .await
            .unwrap();

        // Use dataset's object_store to find and copy the manifest
        let versions_path = dataset.versions_dir();
        let manifest_metas: Vec<_> = dataset
            .object_store(None)
            .await
            .unwrap()
            .inner
            .list(Some(&versions_path))
            .try_collect()
            .await
            .unwrap();

        let manifest_meta = manifest_metas
            .iter()
            .find(|m| {
                m.location
                    .filename()
                    .map(|f| f.ends_with(".manifest"))
                    .unwrap_or(false)
            })
            .expect("No manifest file found");

        // Read the existing manifest data
        let manifest_data = dataset
            .object_store(None)
            .await
            .unwrap()
            .inner
            .get(&manifest_meta.location)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        // Write to a staging location using the dataset's object_store
        let staging_path = dataset.versions_dir().join("staging_manifest");
        dataset
            .object_store(None)
            .await
            .unwrap()
            .inner
            .put(&staging_path, manifest_data.into())
            .await
            .unwrap();

        // Create version 2 from staging manifest
        // Use the same naming scheme as the existing dataset (V2)
        let mut create_version_req = CreateTableVersionRequest::new(2, staging_path.to_string());
        create_version_req.id = Some(table_id.clone());
        create_version_req.naming_scheme = Some("V2".to_string());

        let result = namespace.create_table_version(create_version_req).await;
        assert!(
            result.is_ok(),
            "create_table_version should succeed: {:?}",
            result
        );

        // Verify version 2 was created at the path returned in the response
        let response = result.unwrap();
        let version_info = response
            .version
            .expect("response should contain version info");
        let version_2_path = Path::parse(&version_info.manifest_path).unwrap();
        let head_result = dataset
            .object_store(None)
            .await
            .unwrap()
            .inner
            .head(&version_2_path)
            .await;
        assert!(
            head_result.is_ok(),
            "Version 2 manifest should exist at {}",
            version_2_path
        );

        // Verify the staging file has been deleted
        let staging_head_result = dataset
            .object_store(None)
            .await
            .unwrap()
            .inner
            .head(&staging_path)
            .await;
        assert!(
            staging_head_result.is_err(),
            "Staging manifest should have been deleted after create_table_version"
        );
    }

    #[tokio::test]
    async fn test_create_table_version_conflict() {
        // create_table_version should fail if the version already exists.
        // Each version always writes to a new file location.
        use futures::TryStreamExt;
        use lance::dataset::builder::DatasetBuilder;
        use lance_namespace::models::CreateTableVersionRequest;

        let temp_dir = TempStrDir::default();
        let temp_path: &str = &temp_dir;

        let namespace: Arc<dyn LanceNamespace> = Arc::new(
            DirectoryNamespaceBuilder::new(temp_path)
                .table_version_tracking_enabled(true)
                .build()
                .await
                .unwrap(),
        );

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Open the dataset using from_namespace to get proper object_store and paths
        let table_id = vec!["test_table".to_string()];
        let dataset = DatasetBuilder::from_namespace(namespace.clone(), table_id.clone())
            .await
            .unwrap()
            .load()
            .await
            .unwrap();

        // Use dataset's object_store to find and copy the manifest
        let versions_path = dataset.versions_dir();
        let manifest_metas: Vec<_> = dataset
            .object_store(None)
            .await
            .unwrap()
            .inner
            .list(Some(&versions_path))
            .try_collect()
            .await
            .unwrap();

        let manifest_meta = manifest_metas
            .iter()
            .find(|m| {
                m.location
                    .filename()
                    .map(|f| f.ends_with(".manifest"))
                    .unwrap_or(false)
            })
            .expect("No manifest file found");

        // Read the existing manifest data
        let manifest_data = dataset
            .object_store(None)
            .await
            .unwrap()
            .inner
            .get(&manifest_meta.location)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        // Write to a staging location using the dataset's object_store
        let staging_path = dataset.versions_dir().join("staging_manifest");
        dataset
            .object_store(None)
            .await
            .unwrap()
            .inner
            .put(&staging_path, manifest_data.into())
            .await
            .unwrap();

        // First create version 2 (should succeed)
        let mut create_version_req = CreateTableVersionRequest::new(2, staging_path.to_string());
        create_version_req.id = Some(table_id.clone());
        create_version_req.naming_scheme = Some("V2".to_string());
        let first_result = namespace.create_table_version(create_version_req).await;
        assert!(
            first_result.is_ok(),
            "First create_table_version for version 2 should succeed: {:?}",
            first_result
        );

        // Get the path from the response for verification
        let version_2_path = Path::parse(
            &first_result
                .unwrap()
                .version
                .expect("response should contain version info")
                .manifest_path,
        )
        .unwrap();

        // Create version 2 again (should fail - conflict)
        let mut create_version_req = CreateTableVersionRequest::new(2, staging_path.to_string());
        create_version_req.id = Some(table_id.clone());
        create_version_req.naming_scheme = Some("V2".to_string());

        let result = namespace.create_table_version(create_version_req).await;
        assert!(
            result.is_err(),
            "create_table_version should fail for existing version"
        );

        // Verify version 2 still exists using the dataset's object_store
        let head_result = dataset
            .object_store(None)
            .await
            .unwrap()
            .inner
            .head(&version_2_path)
            .await;
        assert!(
            head_result.is_ok(),
            "Version 2 manifest should still exist at {}",
            version_2_path
        );
    }

    #[tokio::test]
    async fn test_create_table_version_table_not_found() {
        use lance_namespace::models::CreateTableVersionRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap();

        // Try to create version for non-existent table
        let mut create_version_req =
            CreateTableVersionRequest::new(1, "/some/staging/path".to_string());
        create_version_req.id = Some(vec!["non_existent_table".to_string()]);

        let result = namespace.create_table_version(create_version_req).await;
        assert!(
            result.is_err(),
            "create_table_version should fail for non-existent table"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Table not found"),
            "Error should mention table not found, got: {}",
            err_msg
        );
    }

    /// End-to-end integration test module for table version tracking.
    mod e2e_table_version_tracking {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Tracking wrapper around a namespace that counts method invocations.
        struct TrackingNamespace {
            inner: DirectoryNamespace,
            create_table_version_count: AtomicUsize,
            describe_table_version_count: AtomicUsize,
            list_table_versions_count: AtomicUsize,
        }

        impl TrackingNamespace {
            fn new(inner: DirectoryNamespace) -> Self {
                Self {
                    inner,
                    create_table_version_count: AtomicUsize::new(0),
                    describe_table_version_count: AtomicUsize::new(0),
                    list_table_versions_count: AtomicUsize::new(0),
                }
            }

            fn create_table_version_calls(&self) -> usize {
                self.create_table_version_count.load(Ordering::SeqCst)
            }

            fn describe_table_version_calls(&self) -> usize {
                self.describe_table_version_count.load(Ordering::SeqCst)
            }

            fn list_table_versions_calls(&self) -> usize {
                self.list_table_versions_count.load(Ordering::SeqCst)
            }
        }

        impl std::fmt::Debug for TrackingNamespace {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("TrackingNamespace")
                    .field(
                        "create_table_version_calls",
                        &self.create_table_version_calls(),
                    )
                    .finish()
            }
        }

        #[async_trait]
        impl LanceNamespace for TrackingNamespace {
            async fn create_namespace(
                &self,
                request: CreateNamespaceRequest,
            ) -> Result<CreateNamespaceResponse> {
                self.inner.create_namespace(request).await
            }

            async fn describe_namespace(
                &self,
                request: DescribeNamespaceRequest,
            ) -> Result<DescribeNamespaceResponse> {
                self.inner.describe_namespace(request).await
            }

            async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
                self.inner.namespace_exists(request).await
            }

            async fn list_namespaces(
                &self,
                request: ListNamespacesRequest,
            ) -> Result<ListNamespacesResponse> {
                self.inner.list_namespaces(request).await
            }

            async fn drop_namespace(
                &self,
                request: DropNamespaceRequest,
            ) -> Result<DropNamespaceResponse> {
                self.inner.drop_namespace(request).await
            }

            async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
                self.inner.list_tables(request).await
            }

            async fn describe_table(
                &self,
                request: DescribeTableRequest,
            ) -> Result<DescribeTableResponse> {
                self.inner.describe_table(request).await
            }

            async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
                self.inner.table_exists(request).await
            }

            async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
                self.inner.drop_table(request).await
            }

            async fn create_table(
                &self,
                request: CreateTableRequest,
                request_data: Bytes,
            ) -> Result<CreateTableResponse> {
                self.inner.create_table(request, request_data).await
            }

            async fn declare_table(
                &self,
                request: DeclareTableRequest,
            ) -> Result<DeclareTableResponse> {
                self.inner.declare_table(request).await
            }

            async fn list_table_versions(
                &self,
                request: ListTableVersionsRequest,
            ) -> Result<ListTableVersionsResponse> {
                self.list_table_versions_count
                    .fetch_add(1, Ordering::SeqCst);
                self.inner.list_table_versions(request).await
            }

            async fn create_table_version(
                &self,
                request: CreateTableVersionRequest,
            ) -> Result<CreateTableVersionResponse> {
                self.create_table_version_count
                    .fetch_add(1, Ordering::SeqCst);
                self.inner.create_table_version(request).await
            }

            async fn describe_table_version(
                &self,
                request: DescribeTableVersionRequest,
            ) -> Result<DescribeTableVersionResponse> {
                self.describe_table_version_count
                    .fetch_add(1, Ordering::SeqCst);
                self.inner.describe_table_version(request).await
            }

            async fn batch_delete_table_versions(
                &self,
                request: BatchDeleteTableVersionsRequest,
            ) -> Result<BatchDeleteTableVersionsResponse> {
                self.inner.batch_delete_table_versions(request).await
            }

            fn namespace_id(&self) -> String {
                self.inner.namespace_id()
            }
        }

        #[tokio::test]
        async fn test_describe_table_returns_managed_versioning() {
            use lance_namespace::models::{CreateNamespaceRequest, DescribeTableRequest};

            let temp_dir = TempStdDir::default();
            let temp_path = temp_dir.to_str().unwrap();

            // Create namespace with table_version_tracking_enabled and manifest_enabled
            let ns = DirectoryNamespaceBuilder::new(temp_path)
                .table_version_tracking_enabled(true)
                .manifest_enabled(true)
                .build()
                .await
                .unwrap();

            // Create parent namespace
            let mut create_ns_req = CreateNamespaceRequest::new();
            create_ns_req.id = Some(vec!["workspace".to_string()]);
            ns.create_namespace(create_ns_req).await.unwrap();

            // Create a table with multi-level ID (namespace + table)
            let schema = create_test_schema();
            let ipc_data = create_test_ipc_data(&schema);
            let mut create_req = CreateTableRequest::new();
            create_req.id = Some(vec!["workspace".to_string(), "test_table".to_string()]);
            ns.create_table(create_req, bytes::Bytes::from(ipc_data))
                .await
                .unwrap();

            // Describe table should return managed_versioning=true
            let mut describe_req = DescribeTableRequest::new();
            describe_req.id = Some(vec!["workspace".to_string(), "test_table".to_string()]);
            let describe_resp = ns.describe_table(describe_req).await.unwrap();

            // managed_versioning should be true
            assert_eq!(
                describe_resp.managed_versioning,
                Some(true),
                "managed_versioning should be true when table_version_tracking_enabled=true"
            );
        }

        #[tokio::test]
        async fn test_external_manifest_store_invokes_namespace_apis() {
            use arrow::array::{Int32Array, StringArray};
            use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
            use arrow::record_batch::RecordBatch;
            use lance::Dataset;
            use lance::dataset::builder::DatasetBuilder;
            use lance::dataset::{WriteMode, WriteParams};
            use lance_namespace::models::CreateNamespaceRequest;

            let temp_dir = TempStdDir::default();
            let temp_path = temp_dir.to_str().unwrap();

            // Create namespace with table_version_tracking_enabled and manifest_enabled
            let inner_ns = DirectoryNamespaceBuilder::new(temp_path)
                .table_version_tracking_enabled(true)
                .manifest_enabled(true)
                .build()
                .await
                .unwrap();

            let tracking_ns = Arc::new(TrackingNamespace::new(inner_ns));
            let ns: Arc<dyn LanceNamespace> = tracking_ns.clone();

            // Create parent namespace
            let mut create_ns_req = CreateNamespaceRequest::new();
            create_ns_req.id = Some(vec!["workspace".to_string()]);
            ns.create_namespace(create_ns_req).await.unwrap();

            // Create a table with multi-level ID (namespace + table)
            let table_id = vec!["workspace".to_string(), "test_table".to_string()];

            // Create some initial data
            let arrow_schema = Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, true),
            ]));
            let batch = RecordBatch::try_new(
                arrow_schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![1, 2, 3])),
                    Arc::new(StringArray::from(vec!["a", "b", "c"])),
                ],
            )
            .unwrap();

            // Create a table using write_into_namespace
            let batches = RecordBatchIterator::new(vec![Ok(batch.clone())], arrow_schema.clone());
            let write_params = WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            };
            let mut dataset = Dataset::write_into_namespace(
                batches,
                ns.clone(),
                table_id.clone(),
                Some(write_params),
            )
            .await
            .unwrap();
            assert_eq!(dataset.version().version, 1);

            // Verify create_table_version was called once during initial write_into_namespace
            assert_eq!(
                tracking_ns.create_table_version_calls(),
                1,
                "create_table_version should have been called once during initial write_into_namespace"
            );

            // Append data - this should call create_table_version again
            let append_batch = RecordBatch::try_new(
                arrow_schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![4, 5, 6])),
                    Arc::new(StringArray::from(vec!["d", "e", "f"])),
                ],
            )
            .unwrap();
            let append_batches = RecordBatchIterator::new(vec![Ok(append_batch)], arrow_schema);
            dataset.append(append_batches, None).await.unwrap();

            assert_eq!(
                tracking_ns.create_table_version_calls(),
                2,
                "create_table_version should have been called twice (once for create, once for append)"
            );

            // checkout_latest should call list_table_versions exactly once
            let initial_list_calls = tracking_ns.list_table_versions_calls();
            let latest_dataset = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
                .await
                .unwrap()
                .load()
                .await
                .unwrap();
            assert_eq!(latest_dataset.version().version, 2);
            assert_eq!(
                tracking_ns.list_table_versions_calls(),
                initial_list_calls + 1,
                "list_table_versions should have been called exactly once during checkout_latest"
            );

            // checkout to specific version should call describe_table_version exactly once
            let initial_describe_calls = tracking_ns.describe_table_version_calls();
            let v1_dataset = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
                .await
                .unwrap()
                .with_version(1)
                .load()
                .await
                .unwrap();
            assert_eq!(v1_dataset.version().version, 1);
            assert_eq!(
                tracking_ns.describe_table_version_calls(),
                initial_describe_calls + 1,
                "describe_table_version should have been called exactly once during checkout to version 1"
            );
        }

        #[tokio::test]
        async fn test_dataset_commit_with_external_manifest_store() {
            use arrow::array::{Int32Array, StringArray};
            use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
            use arrow::record_batch::RecordBatch;
            use futures::TryStreamExt;
            use lance::dataset::{Dataset, WriteMode, WriteParams};
            use lance_namespace::models::CreateNamespaceRequest;
            use lance_table::io::commit::ManifestNamingScheme;

            let temp_dir = TempStdDir::default();
            let temp_path = temp_dir.to_str().unwrap();

            // Create namespace with table_version_tracking_enabled and manifest_enabled
            let inner_ns = DirectoryNamespaceBuilder::new(temp_path)
                .table_version_tracking_enabled(true)
                .manifest_enabled(true)
                .build()
                .await
                .unwrap();

            let tracking_ns: Arc<dyn LanceNamespace> = Arc::new(TrackingNamespace::new(inner_ns));

            // Create parent namespace
            let mut create_ns_req = CreateNamespaceRequest::new();
            create_ns_req.id = Some(vec!["workspace".to_string()]);
            tracking_ns.create_namespace(create_ns_req).await.unwrap();

            // Create a table using write_into_namespace
            let table_id = vec!["workspace".to_string(), "test_table".to_string()];
            let arrow_schema = Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, true),
            ]));
            let batch = RecordBatch::try_new(
                arrow_schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![1, 2, 3])),
                    Arc::new(StringArray::from(vec!["a", "b", "c"])),
                ],
            )
            .unwrap();
            let batches = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema.clone());
            let write_params = WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            };
            let dataset = Dataset::write_into_namespace(
                batches,
                tracking_ns.clone(),
                table_id.clone(),
                Some(write_params),
            )
            .await
            .unwrap();
            assert_eq!(dataset.version().version, 1);

            // Append data using write_into_namespace (APPEND mode)
            let batch2 = RecordBatch::try_new(
                arrow_schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![4, 5, 6])),
                    Arc::new(StringArray::from(vec!["d", "e", "f"])),
                ],
            )
            .unwrap();
            let batches = RecordBatchIterator::new(vec![Ok(batch2)], arrow_schema);
            let write_params = WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            };
            Dataset::write_into_namespace(
                batches,
                tracking_ns.clone(),
                table_id.clone(),
                Some(write_params),
            )
            .await
            .unwrap();

            // Verify version 2 was created using the dataset's object_store
            // List manifests in the versions directory to find the V2 named manifest
            let manifest_metas: Vec<_> = dataset
                .object_store(None)
                .await
                .unwrap()
                .inner
                .list(Some(&dataset.versions_dir()))
                .try_collect()
                .await
                .unwrap();
            let version_2_found = manifest_metas.iter().any(|m| {
                m.location
                    .filename()
                    .map(|f| {
                        f.ends_with(".manifest")
                            && ManifestNamingScheme::V2.parse_version(f) == Some(2)
                    })
                    .unwrap_or(false)
            });
            assert!(
                version_2_found,
                "Version 2 manifest should exist in versions directory"
            );
        }

        /// Helper: create a namespace and a table with some rows, returning (namespace, table_id)
        async fn create_ns_with_table() -> (DirectoryNamespace, TempStdDir, Vec<String>) {
            use arrow::array::{Int32Array, StringArray};
            use arrow::ipc::writer::StreamWriter;

            let (namespace, temp_dir) = create_test_namespace().await;

            let schema = create_test_schema();
            let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
            let arrow_schema = Arc::new(arrow_schema);

            let id_array = Int32Array::from(vec![1, 2, 3]);
            let name_array = StringArray::from(vec!["Alice", "Bob", "Charlie"]);
            let batch = arrow::record_batch::RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(id_array), Arc::new(name_array)],
            )
            .unwrap();

            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            let mut request = CreateTableRequest::new();
            let table_id = vec!["test_ops_table".to_string()];
            request.id = Some(table_id.clone());

            namespace
                .create_table(request, Bytes::from(buffer))
                .await
                .unwrap();

            (namespace, temp_dir, table_id)
        }

        #[tokio::test]
        async fn test_count_table_rows_basic() {
            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            let request = CountTableRowsRequest {
                id: Some(table_id),
                version: None,
                predicate: None,
                ..Default::default()
            };

            let count = namespace.count_table_rows(request).await.unwrap();
            assert_eq!(count, 3);
        }

        #[tokio::test]
        async fn test_count_table_rows_with_predicate() {
            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            let request = CountTableRowsRequest {
                id: Some(table_id),
                version: None,
                predicate: Some("id > 1".to_string()),
                ..Default::default()
            };

            let count = namespace.count_table_rows(request).await.unwrap();
            assert_eq!(count, 2);
        }

        #[tokio::test]
        async fn test_query_table_invalid_distance_type() {
            let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

            let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
                single_vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
                multi_vector: None,
            });

            let request = QueryTableRequest {
                id: Some(table_id),
                k: 2,
                vector,
                vector_column: Some("vector".to_string()),
                distance_type: Some("invalid_metric".to_string()),
                filter: None,
                offset: None,
                version: None,
                ..Default::default()
            };

            let result = namespace.query_table(request).await;
            assert!(result.is_err());
            let err_msg = result.unwrap_err().to_string();
            assert!(
                err_msg.contains("Unknown distance type"),
                "Expected error about unknown distance type, got: {}",
                err_msg
            );
        }

        #[tokio::test]
        async fn test_insert_into_table_append() {
            use arrow::array::{Int32Array, StringArray};
            use arrow::ipc::writer::StreamWriter;

            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            // Prepare new data to insert
            let schema = create_test_schema();
            let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
            let arrow_schema = Arc::new(arrow_schema);

            let id_array = Int32Array::from(vec![4, 5]);
            let name_array = StringArray::from(vec!["Dave", "Eve"]);
            let batch = arrow::record_batch::RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(id_array), Arc::new(name_array)],
            )
            .unwrap();

            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            let request = InsertIntoTableRequest {
                id: Some(table_id.clone()),
                mode: Some("append".to_string()),
                ..Default::default()
            };

            let response = namespace
                .insert_into_table(request, Bytes::from(buffer))
                .await
                .unwrap();
            assert!(response.transaction_id.is_none());

            // Verify total rows
            let count_req = CountTableRowsRequest {
                id: Some(table_id),
                version: None,
                predicate: None,
                ..Default::default()
            };
            let count = namespace.count_table_rows(count_req).await.unwrap();
            assert_eq!(count, 5);
        }

        #[tokio::test]
        async fn test_insert_into_table_overwrite() {
            use arrow::array::{Int32Array, StringArray};
            use arrow::ipc::writer::StreamWriter;

            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            let schema = create_test_schema();
            let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
            let arrow_schema = Arc::new(arrow_schema);

            let id_array = Int32Array::from(vec![10, 20]);
            let name_array = StringArray::from(vec!["X", "Y"]);
            let batch = arrow::record_batch::RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(id_array), Arc::new(name_array)],
            )
            .unwrap();

            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            let request = InsertIntoTableRequest {
                id: Some(table_id.clone()),
                mode: Some("overwrite".to_string()),
                ..Default::default()
            };

            namespace
                .insert_into_table(request, Bytes::from(buffer))
                .await
                .unwrap();

            // Verify overwrite: only 2 rows remain
            let count_req = CountTableRowsRequest {
                id: Some(table_id),
                version: None,
                predicate: None,
                ..Default::default()
            };
            let count = namespace.count_table_rows(count_req).await.unwrap();
            assert_eq!(count, 2);
        }

        #[tokio::test]
        async fn test_insert_into_table_empty_data() {
            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            let request = InsertIntoTableRequest {
                id: Some(table_id),
                mode: None,
                ..Default::default()
            };

            let result = namespace.insert_into_table(request, Bytes::new()).await;
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("Arrow IPC stream) is required")
            );
        }

        #[tokio::test]
        async fn test_insert_into_table_with_storage_options() {
            use arrow::array::{Int32Array, StringArray};
            use arrow::ipc::writer::StreamWriter;

            let temp_dir = TempStdDir::default();

            // Build namespace with a (no-op) storage option so self.storage_options is Some
            let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
                .storage_option("allow_http", "true")
                .build()
                .await
                .unwrap();

            // Create a table first
            let schema = create_test_schema();
            let ipc_data = create_test_ipc_data(&schema);
            let mut create_req = CreateTableRequest::new();
            let table_id = vec!["so_table".to_string()];
            create_req.id = Some(table_id.clone());
            namespace
                .create_table(create_req, Bytes::from(ipc_data))
                .await
                .unwrap();

            // Insert with storage_options present — covers store_params closure
            let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
            let arrow_schema = Arc::new(arrow_schema);

            let id_array = Int32Array::from(vec![10, 20]);
            let name_array = StringArray::from(vec!["X", "Y"]);
            let batch = arrow::record_batch::RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(id_array), Arc::new(name_array)],
            )
            .unwrap();

            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            let request = InsertIntoTableRequest {
                id: Some(table_id.clone()),
                mode: Some("append".to_string()),
                ..Default::default()
            };

            let response = namespace
                .insert_into_table(request, Bytes::from(buffer))
                .await
                .unwrap();
            assert!(response.transaction_id.is_none());

            // Verify rows were inserted
            let count_req = CountTableRowsRequest {
                id: Some(table_id),
                version: None,
                predicate: None,
                ..Default::default()
            };
            let count = namespace.count_table_rows(count_req).await.unwrap();
            assert_eq!(count, 2);
        }

        #[tokio::test]
        async fn test_query_table_basic() {
            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            let request = QueryTableRequest {
                id: Some(table_id),
                k: 10,
                filter: None,
                offset: None,
                version: None,
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();

            // Decode IPC and verify
            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 3);
        }

        #[tokio::test]
        async fn test_query_table_with_filter() {
            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            let request = QueryTableRequest {
                id: Some(table_id),
                k: 10,
                filter: Some("id <= 2".to_string()),
                offset: None,
                version: None,
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 2);
        }

        #[tokio::test]
        async fn test_query_table_with_limit_and_offset() {
            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            let request = QueryTableRequest {
                id: Some(table_id),
                k: 2,
                filter: None,
                offset: Some(1),
                version: None,
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 2);
        }

        #[tokio::test]
        async fn test_query_table_no_limit() {
            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            // k=0 means no limit
            let request = QueryTableRequest {
                id: Some(table_id),
                k: 0,
                filter: None,
                offset: None,
                version: None,
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 3);
        }

        #[tokio::test]
        async fn test_query_table_with_columns() {
            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            let columns = Box::new(lance_namespace::models::QueryTableRequestColumns {
                column_names: Some(vec!["id".to_string()]),
                column_aliases: None,
            });

            let request = QueryTableRequest {
                id: Some(table_id),
                k: 10,
                filter: None,
                offset: None,
                version: None,
                columns: Some(columns),
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let schema = reader.schema();
            assert_eq!(schema.fields().len(), 1);
            assert_eq!(schema.field(0).name(), "id");
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 3);
        }

        #[tokio::test]
        async fn test_count_table_rows_with_version() {
            use arrow::array::{Int32Array, StringArray};
            use arrow::ipc::writer::StreamWriter;

            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            // Insert more data to create version 2
            let schema = create_test_schema();
            let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
            let arrow_schema = Arc::new(arrow_schema);

            let id_array = Int32Array::from(vec![4, 5]);
            let name_array = StringArray::from(vec!["Dave", "Eve"]);
            let batch = arrow::record_batch::RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(id_array), Arc::new(name_array)],
            )
            .unwrap();

            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            let request = InsertIntoTableRequest {
                id: Some(table_id.clone()),
                mode: None,
                ..Default::default()
            };
            namespace
                .insert_into_table(request, Bytes::from(buffer))
                .await
                .unwrap();

            // Version 1 should have 3 rows
            let count_req = CountTableRowsRequest {
                id: Some(table_id.clone()),
                version: Some(1),
                predicate: None,
                ..Default::default()
            };
            let count = namespace.count_table_rows(count_req).await.unwrap();
            assert_eq!(count, 3);

            // Latest version should have 5 rows
            let count_req = CountTableRowsRequest {
                id: Some(table_id),
                version: None,
                predicate: None,
                ..Default::default()
            };
            let count = namespace.count_table_rows(count_req).await.unwrap();
            assert_eq!(count, 5);
        }

        #[tokio::test]
        async fn test_query_table_with_version() {
            use arrow::array::{Int32Array, StringArray};
            use arrow::ipc::writer::StreamWriter;

            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            // Insert more data to create version 2
            let schema = create_test_schema();
            let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
            let arrow_schema = Arc::new(arrow_schema);

            let id_array = Int32Array::from(vec![4, 5]);
            let name_array = StringArray::from(vec!["Dave", "Eve"]);
            let batch = arrow::record_batch::RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(id_array), Arc::new(name_array)],
            )
            .unwrap();

            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            let request = InsertIntoTableRequest {
                id: Some(table_id.clone()),
                mode: None,
                ..Default::default()
            };
            namespace
                .insert_into_table(request, Bytes::from(buffer))
                .await
                .unwrap();

            // Query version 1 should return 3 rows
            let request = QueryTableRequest {
                id: Some(table_id.clone()),
                k: 100,
                filter: None,
                offset: None,
                version: Some(1),
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();
            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 3);

            // Query latest version should return 5 rows
            let request = QueryTableRequest {
                id: Some(table_id),
                k: 100,
                filter: None,
                offset: None,
                version: None,
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();
            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 5);
        }

        /// Helper to create a namespace with a table that has a vector column for
        /// vector search tests.
        async fn create_ns_with_vector_table() -> (DirectoryNamespace, TempStdDir, Vec<String>) {
            use arrow::array::{FixedSizeListArray, Float32Array, Int32Array};
            use arrow::ipc::writer::StreamWriter;

            let (namespace, temp_dir) = create_test_namespace().await;

            // Build schema: id (int32), vector (fixed_size_list<float32>[4])
            let arrow_schema = Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int32, false),
                arrow::datatypes::Field::new(
                    "vector",
                    arrow::datatypes::DataType::FixedSizeList(
                        Arc::new(arrow::datatypes::Field::new(
                            "item",
                            arrow::datatypes::DataType::Float32,
                            true,
                        )),
                        4,
                    ),
                    true,
                ),
            ]));

            let id_array = Int32Array::from(vec![1, 2, 3]);
            let values = Float32Array::from(vec![
                1.0, 0.0, 0.0, 0.0, // vector for id=1
                0.0, 1.0, 0.0, 0.0, // vector for id=2
                0.0, 0.0, 1.0, 0.0, // vector for id=3
            ]);
            let vector_array = FixedSizeListArray::try_new(
                Arc::new(arrow::datatypes::Field::new(
                    "item",
                    arrow::datatypes::DataType::Float32,
                    true,
                )),
                4,
                Arc::new(values),
                None,
            )
            .unwrap();

            let batch = arrow::record_batch::RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(id_array), Arc::new(vector_array)],
            )
            .unwrap();

            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            // Write as a Lance dataset directly
            let table_name = "vector_table";
            let table_uri = format!("{}/{}.lance", temp_dir.to_str().unwrap(), table_name);
            let reader = arrow::record_batch::RecordBatchIterator::new(
                vec![Ok(batch)],
                arrow_schema.clone(),
            );
            Dataset::write(reader, &table_uri, None).await.unwrap();

            let table_id = vec![table_name.to_string()];
            (namespace, temp_dir, table_id)
        }

        #[tokio::test]
        async fn test_query_table_vector_search() {
            let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

            let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
                single_vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
                multi_vector: None,
            });

            let request = QueryTableRequest {
                id: Some(table_id),
                k: 2,
                vector,
                filter: None,
                offset: None,
                version: None,
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 2);
        }

        #[tokio::test]
        async fn test_query_table_vector_search_with_distance_type() {
            let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

            let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
                single_vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
                multi_vector: None,
            });

            let request = QueryTableRequest {
                id: Some(table_id),
                k: 3,
                vector,
                filter: None,
                offset: None,
                version: None,
                distance_type: Some("cosine".to_string()),
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 3);
        }

        #[tokio::test]
        async fn test_query_table_vector_search_with_filter() {
            let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

            let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
                single_vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
                multi_vector: None,
            });

            let request = QueryTableRequest {
                id: Some(table_id),
                k: 10,
                vector,
                filter: Some("id <= 2".to_string()),
                offset: None,
                version: None,
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert!(total_rows <= 2);
        }

        #[tokio::test]
        async fn test_query_table_vector_search_with_nprobes_and_refine() {
            let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

            let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
                single_vector: Some(vec![0.0, 1.0, 0.0, 0.0]),
                multi_vector: None,
            });

            let request = QueryTableRequest {
                id: Some(table_id),
                k: 2,
                vector,
                filter: None,
                offset: None,
                version: None,
                nprobes: Some(1),
                refine_factor: Some(1),
                prefilter: Some(true),
                ..Default::default()
            };

            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 2);
        }

        #[tokio::test]
        async fn test_namespace_id() {
            let (namespace, _temp_dir) = create_test_namespace().await;
            let id = namespace.namespace_id();
            assert!(id.contains("DirectoryNamespace"));
            assert!(id.contains("root"));
        }

        #[tokio::test]
        async fn test_query_table_empty_table() {
            let (namespace, _temp_dir) = create_test_namespace().await;

            // Create table with empty IPC data (schema only, no rows)
            let schema = create_test_schema();
            let ipc_data = create_test_ipc_data(&schema);
            let mut create_request = CreateTableRequest::new();
            create_request.id = Some(vec!["empty_table".to_string()]);
            namespace
                .create_table(create_request, bytes::Bytes::from(ipc_data))
                .await
                .unwrap();

            // Query the empty table — should hit the "no batches" else branch
            let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
                single_vector: None,
                multi_vector: None,
            });
            let request = QueryTableRequest {
                id: Some(vec!["empty_table".to_string()]),
                k: 10,
                vector,
                ..Default::default()
            };
            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.collect::<std::result::Result<Vec<_>, _>>().unwrap();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 0, "empty table should yield no rows");
        }

        #[tokio::test]
        async fn test_query_table_with_plain_filter_no_vector() {
            let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

            // Query with filter but no vector (plain scan path + filter)
            let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
                single_vector: None,
                multi_vector: None,
            });
            let request = QueryTableRequest {
                id: Some(table_id),
                k: 0,
                vector,
                filter: Some("id > 1".to_string()),
                ..Default::default()
            };
            let bytes = namespace.query_table(request).await.unwrap();

            let cursor = Cursor::new(bytes.to_vec());
            let reader = FileReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert!(total_rows > 0);
            assert!(total_rows < 3);
        }
    }

    /// Tests for multi-table transaction support via table_version_storage_enabled.
    mod multi_table_transactions {
        use super::*;
        use futures::TryStreamExt;
        use lance::dataset::builder::DatasetBuilder;
        use lance_namespace::models::CreateTableVersionRequest;

        /// Helper to create a namespace with table_version_storage_enabled enabled
        async fn create_managed_namespace(temp_path: &str) -> Arc<DirectoryNamespace> {
            Arc::new(
                DirectoryNamespaceBuilder::new(temp_path)
                    .table_version_tracking_enabled(true)
                    .table_version_storage_enabled(true)
                    .manifest_enabled(true)
                    .build()
                    .await
                    .unwrap(),
            )
        }

        /// Helper to create a table and get its staging manifest path
        async fn create_table_and_get_staging(
            namespace: Arc<dyn LanceNamespace>,
            table_name: &str,
        ) -> (Vec<String>, object_store::path::Path) {
            let schema = create_test_schema();
            let ipc_data = create_test_ipc_data(&schema);
            let mut create_req = CreateTableRequest::new();
            create_req.id = Some(vec![table_name.to_string()]);
            namespace
                .create_table(create_req, bytes::Bytes::from(ipc_data))
                .await
                .unwrap();

            let table_id = vec![table_name.to_string()];
            let dataset = DatasetBuilder::from_namespace(namespace.clone(), table_id.clone())
                .await
                .unwrap()
                .load()
                .await
                .unwrap();

            // Find existing manifest and create a staging copy
            let versions_path = dataset.versions_dir();
            let manifest_metas: Vec<_> = dataset
                .object_store(None)
                .await
                .unwrap()
                .inner
                .list(Some(&versions_path))
                .try_collect()
                .await
                .unwrap();

            let manifest_meta = manifest_metas
                .iter()
                .find(|m| {
                    m.location
                        .filename()
                        .map(|f| f.ends_with(".manifest"))
                        .unwrap_or(false)
                })
                .expect("No manifest file found");

            let manifest_data = dataset
                .object_store(None)
                .await
                .unwrap()
                .inner
                .get(&manifest_meta.location)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();

            let staging_path = dataset
                .versions_dir()
                .join(format!("staging_{}", table_name));
            dataset
                .object_store(None)
                .await
                .unwrap()
                .inner
                .put(&staging_path, manifest_data.into())
                .await
                .unwrap();

            (table_id, staging_path)
        }

        #[tokio::test]
        async fn test_table_version_storage_enabled_requires_manifest() {
            // table_version_storage_enabled=true requires manifest_enabled=true
            let temp_dir = TempStdDir::default();
            let temp_path = temp_dir.to_str().unwrap();

            let result = DirectoryNamespaceBuilder::new(temp_path)
                .table_version_storage_enabled(true)
                .manifest_enabled(false)
                .build()
                .await;

            assert!(
                result.is_err(),
                "Should fail when table_version_storage_enabled=true but manifest_enabled=false"
            );
        }

        #[tokio::test]
        async fn test_create_table_version_records_in_manifest() {
            // When table_version_storage_enabled is enabled, single create_table_version
            // should also record the version in __manifest
            let temp_dir = TempStrDir::default();
            let temp_path: &str = &temp_dir;

            let namespace = create_managed_namespace(temp_path).await;
            let ns: Arc<dyn LanceNamespace> = namespace.clone();

            let (table_id, staging_path) =
                create_table_and_get_staging(ns.clone(), "table_managed").await;

            // Create version 2
            let mut create_req = CreateTableVersionRequest::new(2, staging_path.to_string());
            create_req.id = Some(table_id.clone());
            create_req.naming_scheme = Some("V2".to_string());
            let response = namespace.create_table_version(create_req).await.unwrap();

            assert!(response.version.is_some());
            let version = response.version.unwrap();
            assert_eq!(version.version, 2);

            // Verify the version is recorded in __manifest by listing versions
            // through the same production path used by the namespace.
            let manifest_ns = namespace.manifest_ns.as_ref().unwrap();
            let versions = manifest_ns
                .list_table_versions(&table_id, false, None)
                .await
                .unwrap();

            assert!(
                !versions.versions.is_empty(),
                "Version should be recorded in __manifest"
            );
            assert_eq!(
                versions.versions[0].version, 2,
                "Recorded version should be 2"
            );
        }
    }

    #[tokio::test]
    async fn test_list_all_tables() {
        use lance_namespace::models::ListTablesRequest;

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "alpha").await;
        create_scalar_table(&namespace, "beta").await;

        let request = ListTablesRequest {
            id: Some(vec![]),
            page_token: None,
            limit: None,
            ..Default::default()
        };
        let response = namespace.list_all_tables(request).await.unwrap();
        let mut tables = response.tables;
        tables.sort();
        assert_eq!(tables, vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn test_restore_table() {
        use lance_namespace::models::RestoreTableRequest;

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "users").await;

        // Create a second version by creating a scalar index (this adds a new version)
        create_scalar_index(&namespace, "users", "users_id_idx").await;

        let dataset = open_dataset(&namespace, "users").await;
        let current_version = dataset.version().version;
        assert!(current_version >= 2, "Should have at least 2 versions");

        // Restore to version 1
        let mut restore_req = RestoreTableRequest::new(1);
        restore_req.id = Some(vec!["users".to_string()]);
        let response = namespace.restore_table(restore_req).await.unwrap();

        // transaction_id should be present (the restore operation)
        assert!(
            response.transaction_id.is_some(),
            "restore_table should return a transaction_id"
        );

        // Verify the dataset now has a new version (restore creates a new version)
        let dataset_after = open_dataset(&namespace, "users").await;
        assert!(
            dataset_after.version().version > current_version,
            "Restore should create a new version"
        );
    }

    #[tokio::test]
    async fn test_update_table_schema_metadata() {
        use lance_namespace::models::UpdateTableSchemaMetadataRequest;

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "products").await;

        let mut metadata = HashMap::new();
        metadata.insert("owner".to_string(), "team_a".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        let mut req = UpdateTableSchemaMetadataRequest::new();
        req.id = Some(vec!["products".to_string()]);
        req.metadata = Some(metadata.clone());

        let response = namespace.update_table_schema_metadata(req).await.unwrap();

        assert!(response.metadata.is_some());
        let returned = response.metadata.unwrap();
        assert_eq!(returned.get("owner"), Some(&"team_a".to_string()));
        assert_eq!(returned.get("version"), Some(&"1.0".to_string()));
        assert!(
            response.transaction_id.is_some(),
            "update_table_schema_metadata should return a transaction_id"
        );
    }

    #[tokio::test]
    async fn test_get_table_stats() {
        use lance_namespace::models::GetTableStatsRequest;

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "items").await;
        create_scalar_index(&namespace, "items", "items_id_idx").await;

        let mut req = GetTableStatsRequest::new();
        req.id = Some(vec!["items".to_string()]);

        let response = namespace.get_table_stats(req).await.unwrap();
        assert_eq!(response.num_rows, 3);
        assert_eq!(response.num_indices, 1);
    }

    #[tokio::test]
    async fn test_explain_table_query_plan() {
        use lance_namespace::models::QueryTableRequestVector;
        use lance_namespace::models::{ExplainTableQueryPlanRequest, QueryTableRequest};

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "catalog").await;

        let mut query = QueryTableRequest::new(1, QueryTableRequestVector::new());
        query.filter = Some("id > 1".to_string());
        query.columns = Some(Box::new(QueryTableRequestColumns {
            column_names: Some(vec!["id".to_string(), "name".to_string()]),
            column_aliases: None,
        }));
        query.with_row_id = Some(true);

        let mut req = ExplainTableQueryPlanRequest::new(query);
        req.id = Some(vec!["catalog".to_string()]);

        let plan_str = namespace.explain_table_query_plan(req).await.unwrap();
        assert_plan_contains_all(
            &plan_str,
            &[
                "ProjectionExec: expr=[id@0 as id, name@2 as name",
                "Take: columns=\"id, _rowid, (name)\"",
                "LanceRead: uri=",
                "projection=[id]",
                "row_id=true, row_addr=false",
                "full_filter=id > Int32(1)",
                "refine_filter=id > Int32(1)",
            ],
            "Filtered explain plan should preserve late materialization and filter pushdown",
        );
    }

    #[tokio::test]
    async fn test_analyze_table_query_plan() {
        use lance_namespace::models::AnalyzeTableQueryPlanRequest;
        use lance_namespace::models::QueryTableRequestVector;

        let (namespace, _temp_dir) = create_test_namespace().await;
        create_scalar_table(&namespace, "catalog").await;

        let mut req = AnalyzeTableQueryPlanRequest::new(1, QueryTableRequestVector::new());
        req.id = Some(vec!["catalog".to_string()]);
        req.filter = Some("id > 0".to_string());
        req.columns = Some(Box::new(QueryTableRequestColumns {
            column_names: Some(vec!["id".to_string(), "name".to_string()]),
            column_aliases: None,
        }));
        req.with_row_id = Some(true);

        let analysis_str = namespace.analyze_table_query_plan(req).await.unwrap();
        assert_plan_contains_all(
            &analysis_str,
            &[
                "AnalyzeExec verbose=true",
                "ProjectionExec: elapsed=",
                "expr=[id@0 as id, name@2 as name",
                "Take: elapsed=",
                "columns=\"id, _rowid, (name)\"",
                "CoalesceBatchesExec: elapsed=",
                "LanceRead: elapsed=",
                "projection=[id]",
                "row_id=true, row_addr=false",
                "full_filter=id > Int32(0)",
                "refine_filter=id > Int32(0)",
                "metrics=[output_rows=",
            ],
            "Filtered analyze plan should preserve late materialization and filter pushdown",
        );
    }

    #[tokio::test]
    async fn test_dir_listing_no_extra_calls_without_migration() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let root_uri = file_object_store_uri(temp_path);
        let listing_count = Arc::new(AtomicUsize::new(0));
        let session = build_listing_counting_session(listing_count.clone());

        // Create a table using dir-listing-only namespace
        let dir_only_ns = DirectoryNamespaceBuilder::new(root_uri.clone())
            .session(session.clone())
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        dir_only_ns
            .create_table(create_req, Bytes::from(ipc_data))
            .await
            .unwrap();

        // Build a namespace with both enabled but migration disabled (default)
        let hybrid_ns = DirectoryNamespaceBuilder::new(root_uri)
            .session(session)
            .manifest_enabled(true)
            .dir_listing_enabled(true)
            .dir_listing_to_manifest_migration_enabled(false)
            .build()
            .await
            .unwrap();

        // Reset counter before the operation we want to measure
        listing_count.store(0, Ordering::SeqCst);

        // table_exists should use dir listing directly, making only 1 listing call
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        hybrid_ns.table_exists(exists_req).await.unwrap();

        let count = listing_count.load(Ordering::SeqCst);
        assert_eq!(
            count, 1,
            "Expected exactly 1 listing call for table_exists \
             without migration mode, but got {}",
            count
        );

        // Reset and test describe_table
        listing_count.store(0, Ordering::SeqCst);

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        hybrid_ns.describe_table(describe_req).await.unwrap();

        let count = listing_count.load(Ordering::SeqCst);
        assert_eq!(
            count, 1,
            "Expected exactly 1 listing call for describe_table \
             without migration mode, but got {}",
            count
        );
    }

    #[tokio::test]
    async fn test_describe_declared_table_checks_versions_only_when_requested() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let root_uri = file_object_store_uri(temp_path);
        let listing_count = Arc::new(AtomicUsize::new(0));
        let session = build_listing_counting_session(listing_count.clone());

        let namespace = DirectoryNamespaceBuilder::new(root_uri)
            .session(session)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        namespace.declare_table(declare_req).await.unwrap();

        listing_count.store(0, Ordering::SeqCst);

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();

        assert_eq!(describe_response.is_only_declared, None);
        assert_eq!(
            listing_count.load(Ordering::SeqCst),
            1,
            "Default describe_table should only list the table directory"
        );

        listing_count.store(0, Ordering::SeqCst);

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        describe_req.check_declared = Some(true);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();

        assert_eq!(describe_response.is_only_declared, Some(true));
        assert_eq!(
            listing_count.load(Ordering::SeqCst),
            2,
            "check_declared describe_table should list the table directory and _versions"
        );
    }

    #[tokio::test]
    async fn test_dir_listing_extra_calls_with_migration() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let root_uri = file_object_store_uri(temp_path);
        let listing_count = Arc::new(AtomicUsize::new(0));
        let session = build_listing_counting_session(listing_count.clone());

        // Create a table using dir-listing-only namespace so it exists physically but is absent from __manifest.
        let dir_only_ns = DirectoryNamespaceBuilder::new(root_uri.clone())
            .session(session.clone())
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        dir_only_ns
            .create_table(create_req, Bytes::from(ipc_data))
            .await
            .unwrap();

        let hybrid_ns = DirectoryNamespaceBuilder::new(root_uri)
            .session(session)
            .manifest_enabled(true)
            .dir_listing_enabled(true)
            .dir_listing_to_manifest_migration_enabled(true)
            .build()
            .await
            .unwrap();

        // table_exists first checks __manifest (which on local FS uses the
        // version hint and does no list call), then falls back to the table
        // directory (one list_with_delimiter on test_table.lance).
        listing_count.store(0, Ordering::SeqCst);

        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        hybrid_ns.table_exists(exists_req).await.unwrap();

        let count = listing_count.load(Ordering::SeqCst);
        assert_eq!(
            count, 1,
            "Expected exactly 1 listing call for table_exists with migration mode \
             (table directory fallback; manifest reload uses the version hint), but got {}",
            count
        );

        // describe_table follows the same path when the table is not yet registered in __manifest.
        listing_count.store(0, Ordering::SeqCst);

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        hybrid_ns.describe_table(describe_req).await.unwrap();

        let count = listing_count.load(Ordering::SeqCst);
        assert_eq!(
            count, 1,
            "Expected exactly 1 listing call for describe_table with migration mode \
             (table directory fallback; manifest reload uses the version hint), but got {}",
            count
        );
    }

    #[tokio::test]
    async fn test_migration_not_found_errors_include_table_id() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(true)
            .dir_listing_to_manifest_migration_enabled(true)
            .build()
            .await
            .unwrap();

        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["missing_table".to_string()]);
        let err = namespace.table_exists(exists_req).await.unwrap_err();
        assert!(matches!(err, Error::Namespace { .. }));
        let err_msg = err.to_string();
        assert!(err_msg.contains("Table not found"));
        assert!(err_msg.contains("table id 'missing_table'"));

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["missing_table".to_string()]);
        let err = namespace.describe_table(describe_req).await.unwrap_err();
        assert!(matches!(err, Error::Namespace { .. }));
        let err_msg = err.to_string();
        assert!(err_msg.contains("Table not found"));
        assert!(err_msg.contains("table id 'missing_table'"));
    }

    #[tokio::test]
    async fn test_manifest_not_found_errors_include_full_table_id() {
        use lance_namespace::models::CreateNamespaceRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["workspace".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        let missing_table_id = vec!["workspace".to_string(), "missing_table".to_string()];

        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(missing_table_id.clone());
        let err = namespace.table_exists(exists_req).await.unwrap_err();
        assert!(matches!(err, Error::Namespace { .. }));
        let err_msg = err.to_string();
        assert!(err_msg.contains("Table not found"));
        assert!(err_msg.contains("table id 'workspace$missing_table'"));

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(missing_table_id);
        let err = namespace.describe_table(describe_req).await.unwrap_err();
        assert!(matches!(err, Error::Namespace { .. }));
        let err_msg = err.to_string();
        assert!(err_msg.contains("Table not found"));
        assert!(err_msg.contains("table id 'workspace$missing_table'"));
    }

    /// Helper used by tag tests: creates a table with `versions` total versions
    /// (1 create + N-1 appends) and returns the namespace plus the table id.
    async fn create_tagged_test_table(
        versions: u32,
    ) -> (Arc<DirectoryNamespace>, TempStdDir, Vec<String>) {
        use arrow::array::{Int32Array, RecordBatchIterator};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use lance::dataset::{Dataset, WriteMode, WriteParams};

        assert!(versions >= 1, "versions must be at least 1");

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = Arc::new(
            DirectoryNamespaceBuilder::new(temp_path)
                .build()
                .await
                .unwrap(),
        );
        let table_id = vec!["tag_table".to_string()];
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let initial_batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(initial_batch)], arrow_schema.clone());
        let write_params = WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        };

        let mut dataset = Dataset::write_into_namespace(
            batches,
            namespace.clone() as Arc<dyn LanceNamespace>,
            table_id.clone(),
            Some(write_params),
        )
        .await
        .unwrap();

        for i in 1..versions {
            let value_start = (i as i32) * 10;
            let batch = RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(Int32Array::from(vec![
                    value_start,
                    value_start + 1,
                ]))],
            )
            .unwrap();
            let batches = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema.clone());
            dataset.append(batches, None).await.unwrap();
        }

        (namespace, temp_dir, table_id)
    }

    #[tokio::test]
    async fn test_create_and_list_tags() {
        let (namespace, _temp_dir, table_id) = create_tagged_test_table(3).await;

        let mut req = CreateTableTagRequest::new("v1".to_string(), 1);
        req.id = Some(table_id.clone());
        namespace.create_table_tag(req).await.unwrap();

        let mut req = CreateTableTagRequest::new("v2".to_string(), 2);
        req.id = Some(table_id.clone());
        namespace.create_table_tag(req).await.unwrap();

        let mut list_req = ListTableTagsRequest::new();
        list_req.id = Some(table_id);
        let resp = namespace.list_table_tags(list_req).await.unwrap();

        assert_eq!(resp.tags.len(), 2, "expected 2 tags, got: {:?}", resp.tags);
        assert_eq!(resp.tags.get("v1").unwrap().version, 1);
        assert_eq!(resp.tags.get("v2").unwrap().version, 2);
        assert!(resp.page_token.is_none());
    }

    #[tokio::test]
    async fn test_create_existing_tag_conflict() {
        let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

        let mut req = CreateTableTagRequest::new("v1".to_string(), 1);
        req.id = Some(table_id.clone());
        namespace.create_table_tag(req).await.unwrap();

        let mut req = CreateTableTagRequest::new("v1".to_string(), 2);
        req.id = Some(table_id);
        let err = namespace.create_table_tag(req).await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("already exists"),
            "expected already-exists error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_get_tag_version() {
        let (namespace, _temp_dir, table_id) = create_tagged_test_table(3).await;

        let mut req = CreateTableTagRequest::new("release".to_string(), 2);
        req.id = Some(table_id.clone());
        namespace.create_table_tag(req).await.unwrap();

        let mut get_req = GetTableTagVersionRequest::new("release".to_string());
        get_req.id = Some(table_id);
        let resp = namespace.get_table_tag_version(get_req).await.unwrap();
        assert_eq!(resp.version, 2);
    }

    #[tokio::test]
    async fn test_get_unknown_tag() {
        let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

        let mut get_req = GetTableTagVersionRequest::new("does-not-exist".to_string());
        get_req.id = Some(table_id);
        let err = namespace.get_table_tag_version(get_req).await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("not found"),
            "expected not-found error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_update_tag_to_new_version() {
        let (namespace, _temp_dir, table_id) = create_tagged_test_table(3).await;

        let mut req = CreateTableTagRequest::new("rolling".to_string(), 1);
        req.id = Some(table_id.clone());
        namespace.create_table_tag(req).await.unwrap();

        let mut update_req = UpdateTableTagRequest::new("rolling".to_string(), 3);
        update_req.id = Some(table_id.clone());
        namespace.update_table_tag(update_req).await.unwrap();

        let mut get_req = GetTableTagVersionRequest::new("rolling".to_string());
        get_req.id = Some(table_id);
        let resp = namespace.get_table_tag_version(get_req).await.unwrap();
        assert_eq!(resp.version, 3);
    }

    #[tokio::test]
    async fn test_update_unknown_tag() {
        let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

        let mut update_req = UpdateTableTagRequest::new("ghost".to_string(), 1);
        update_req.id = Some(table_id);
        let err = namespace.update_table_tag(update_req).await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("not found"),
            "expected not-found error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_delete_tag() {
        let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

        let mut req = CreateTableTagRequest::new("doomed".to_string(), 1);
        req.id = Some(table_id.clone());
        namespace.create_table_tag(req).await.unwrap();

        let mut delete_req = DeleteTableTagRequest::new("doomed".to_string());
        delete_req.id = Some(table_id.clone());
        namespace.delete_table_tag(delete_req).await.unwrap();

        let mut list_req = ListTableTagsRequest::new();
        list_req.id = Some(table_id.clone());
        let resp = namespace.list_table_tags(list_req).await.unwrap();
        assert!(resp.tags.is_empty(), "tag should be removed after delete");

        // A second get should return NotFound.
        let mut get_req = GetTableTagVersionRequest::new("doomed".to_string());
        get_req.id = Some(table_id);
        let err = namespace.get_table_tag_version(get_req).await.unwrap_err();
        assert!(err.to_string().to_lowercase().contains("not found"));
    }

    #[tokio::test]
    async fn test_delete_unknown_tag() {
        let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

        let mut delete_req = DeleteTableTagRequest::new("nope".to_string());
        delete_req.id = Some(table_id);
        let err = namespace.delete_table_tag(delete_req).await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("not found"),
            "expected not-found error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_create_tag_invalid_version() {
        let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

        // version 0 should be rejected as InvalidInput before reaching the dataset.
        let mut req = CreateTableTagRequest::new("v0".to_string(), 0);
        req.id = Some(table_id.clone());
        let err = namespace.create_table_tag(req).await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("positive"),
            "expected positive-version error, got: {}",
            err
        );

        // empty tag name should also be rejected.
        let mut req = CreateTableTagRequest::new(String::new(), 1);
        req.id = Some(table_id);
        let err = namespace.create_table_tag(req).await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("must not be empty"),
            "expected empty-tag-name error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_create_tag_table_not_found() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let mut req = CreateTableTagRequest::new("v1".to_string(), 1);
        req.id = Some(vec!["does_not_exist".to_string()]);
        let err = namespace.create_table_tag(req).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Table") && msg.to_lowercase().contains("not found"),
            "expected TableNotFound error, got: {}",
            err
        );
    }
}
