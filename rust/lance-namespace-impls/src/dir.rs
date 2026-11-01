// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Directory-based Lance Namespace implementation.
//!
//! This module provides a directory-based implementation of the Lance namespace
//! that stores tables as Lance datasets in a filesystem directory structure.

pub mod manifest;
pub mod manifest_feature_flags;

mod builder;
mod index_params;
mod load;
mod marker;
mod namespace;
mod query;
mod resolve;
mod transactions;
mod versions;

pub use builder::DirectoryNamespaceBuilder;
pub(crate) use marker::{MarkerFileError, put_marker_file_atomic};

use lance::session::Session;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;

use crate::context::DynamicContextProvider;

use lance_core::{Error, Result};
use lance_namespace::LanceNamespace;

use crate::credentials::CredentialVendor;

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

/// Build SQL expression list for the add_columns operation.
/// Returns an explicit error when the expression is missing, instead of silently using an empty string.
pub(crate) fn build_sql_expressions(
    new_columns: &[lance_namespace::models::AddColumnsEntry],
) -> Result<Vec<(String, String)>> {
    new_columns
        .iter()
        .map(|col| {
            // expression is Option<Option<String>>: outer Option means whether the
            // field is present, inner Option means whether the value is JSON null.
            let expression = col.expression.clone().and_then(|opt| opt).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Expression is required for new column '{}'",
                    col.name
                ))
            })?;
            Ok((col.name.clone(), expression))
        })
        .collect()
}

/// Build column alteration list for the alter_columns operation.
/// Returns an explicit error when data_type conversion fails, instead of silently ignoring it.
pub(crate) fn build_column_alterations(
    alterations: &[lance_namespace::models::AlterColumnsEntry],
) -> Result<Vec<lance::dataset::ColumnAlteration>> {
    alterations
        .iter()
        .map(|entry| {
            let mut alteration = lance::dataset::ColumnAlteration::new(entry.path.clone());
            // rename is Option<Option<String>>: flatten to get the actual rename value.
            if let Some(Some(rename)) = &entry.rename {
                alteration = alteration.rename(rename.clone());
            }
            // nullable is Option<Option<bool>>: flatten to get the actual nullable value.
            if let Some(Some(nullable)) = entry.nullable {
                alteration = alteration.set_nullable(nullable);
            }
            // data_type is Option<serde_json::Value>: only process when present and not null.
            if let Some(data_type) = &entry.data_type
                && !data_type.is_null()
            {
                let type_str = data_type.as_str().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "data_type for column '{}' must be a JSON string, got: {}",
                        entry.path, data_type
                    ))
                })?;
                let json_type =
                    lance_namespace::models::JsonArrowDataType::new(type_str.to_string());
                let dt =
                    lance_namespace::schema::convert_json_arrow_type(&json_type).map_err(|e| {
                        Error::invalid_input(format!(
                            "Failed to parse data_type '{}' for column '{}': {}",
                            type_str, entry.path, e
                        ))
                    })?;
                alteration = alteration.cast_to(dt);
            }
            Ok(alteration)
        })
        .collect()
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

/// Directory-based implementation of Lance Namespace.
///
/// This implementation stores tables as Lance datasets in a directory structure.
/// It supports local filesystems and cloud storage backends through Lance's object store.
///
/// ## Manifest-based Listing
///
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
    manifest_ns: OnceCell<Arc<manifest::ManifestNamespace>>,
    write_manifest_ns: OnceCell<Arc<manifest::ManifestNamespace>>,
    manifest_enabled: bool,
    dir_listing_enabled: bool,
    inline_optimization_enabled: bool,
    commit_retries: Option<u32>,
    /// When true, root-level table operations check the manifest first before
    /// falling back to directory listing. When false, root-level tables skip
    /// the manifest check and use directory listing directly.
    dir_listing_to_manifest_migration_enabled: bool,
    /// When true, `describe_table` returns `managed_versioning: true` to indicate
    /// commits should go through namespace table version APIs.
    table_version_tracking_enabled: bool,
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

impl DirectoryNamespace {
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

    /// Increment the counter for an operation.
    fn record_op(&self, operation: &str) {
        if let Some(ref metrics) = self.ops_metrics {
            metrics.increment(operation);
        }
    }
}

#[cfg(test)]
mod tests;
