// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance::dataset::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance_core::Result;
use lance_namespace::LanceNamespace;
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{DescribeTableRequest, DescribeTableResponse};

use super::{DirectoryNamespace, manifest_feature_flags, snapshot::SnapshotReader};

impl DirectoryNamespace {
    /// Internal describe_table implementation that doesn't record metrics.
    /// Used by both the public describe_table (which records metrics) and
    /// internal callers like resolve_table_location (which shouldn't).
    pub(super) async fn describe_table_impl(
        &self,
        request: DescribeTableRequest,
    ) -> Result<DescribeTableResponse> {
        let is_root_level = request.id.as_ref().is_some_and(|id| id.len() == 1);
        let is_child_table = request.id.as_ref().is_some_and(|id| id.len() > 1);
        let probe_manifest_for_root = self.table_discovery_mode.uses_manifest_for_root();
        let skip_manifest_for_root = is_root_level
            && self.table_discovery_mode.has_directory_listing()
            && !probe_manifest_for_root;
        // Self-heal the manifest wherever it can be authoritative: a child table
        // (no dir-listing fallback), a manifest-only namespace, or migration mode
        // (manifest-first -- it can hold registered_table -> external .lance
        // aliases that dir-listing cannot resolve, so a reader built before
        // __manifest must re-probe to see them). The bypass -- skipping the probe
        // -- applies ONLY to directory-backed root reads without migration,
        // which are served entirely from the directory listing.
        if is_child_table || probe_manifest_for_root {
            self.ensure_read_manifest().await?;
        }
        if let Some(manifest_ns) = self.manifest_ns_for_read()
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
                Err(e) if manifest_feature_flags::is_incompatible_manifest_error(&e) => {
                    // An incompatible manifest must surface "please upgrade"
                    // rather than degrading to a directory-listing view.
                    return Err(e);
                }
                Err(e) if self.table_discovery_mode.has_directory_listing() && is_root_level => {
                    // Only a genuinely-absent table (e.g. an unmigrated on-disk
                    // table) may fall through to the directory check; any other
                    // manifest error must propagate rather than be read as missing.
                    if !Self::is_manifest_table_absent_error(&e) {
                        return Err(Self::classify_storage_error(e));
                    }
                }
                Err(e) => return Err(e),
            }
        }
        if is_child_table {
            return Err(self.child_namespace_requires_manifest_error());
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_id = Self::format_table_id_from_request(&request.id);
        if !self.table_discovery_mode.has_directory_listing() {
            return Err(NamespaceError::TableNotFound { message: table_id }.into());
        }

        let table_uri = self.table_full_uri(&table_name);

        // Atomically check table existence and deregistration status
        let status = self.check_table_status(&table_name).await?;

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

        let snapshot = self
            .snapshot_reader()
            .describe(&request, &table_name, &table_uri, is_only_declared)
            .await?;
        let storage_options = self
            .get_storage_options_for_table(&table_uri, vend_credentials, identity)
            .await?;
        Ok(DescribeTableResponse {
            table: Some(table_name),
            namespace: request
                .id
                .as_ref()
                .map(|id| id[..id.len().saturating_sub(1)].to_vec()),
            location: Some(table_uri.clone()),
            table_uri: Some(table_uri),
            storage_options,
            is_only_declared,
            managed_versioning: self.table_version_tracking_enabled.then_some(true),
            ..snapshot
        })
    }

    fn snapshot_reader(&self) -> SnapshotReader<'_> {
        SnapshotReader {
            table_path: &self.base_path,
            storage_options: self.storage_options.as_ref(),
            session: self.session.as_ref(),
        }
    }

    /// Build a `DatasetBuilder` for `table_uri` with this namespace's storage
    /// options and session applied. Callers add version/branch scoping.
    pub(super) fn configured_builder(&self, table_uri: &str) -> DatasetBuilder {
        self.snapshot_reader().configured_builder(table_uri)
    }

    pub(super) async fn load_dataset(
        &self,
        table_uri: &str,
        version: Option<i64>,
        operation: &str,
    ) -> Result<Dataset> {
        self.load_branch_dataset(table_uri, None, version, operation)
            .await
    }

    /// Like [`Self::load_dataset`], on `branch` when it names one.
    pub(super) async fn load_branch_dataset(
        &self,
        table_uri: &str,
        branch: Option<&str>,
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

        let dataset = match Self::normalized_branch(branch)? {
            Some(branch) => self.open_validated_branch(table_uri, branch).await?,
            None => self
                .configured_builder(table_uri)
                .load()
                .await
                .map_err(|e| {
                    let message = format!(
                        "Failed to open table at '{}' for {}: {}",
                        table_uri, operation, e
                    );
                    Self::map_open_error(e, NamespaceError::TableNotFound { message })
                })?,
        };

        if let Some(version) = version {
            return dataset.checkout_version(version as u64).await.map_err(|e| {
                let message = format!(
                    "Failed to checkout version {} for table at '{}' during {}: {}",
                    version, table_uri, operation, e
                );
                Self::map_open_error(e, NamespaceError::TableVersionNotFound { message })
            });
        }

        Ok(dataset)
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
    /// #![recursion_limit = "256"]
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
        let Some(manifest_ns) = self.manifest_ns_for_write().await? else {
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
}
