// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use lance::dataset::statistics::DatasetStatisticsExt;
use lance::dataset::{
    Dataset, MergeInsertBuilder, UpdateBuilder, WhenMatched, WhenNotMatched,
    WhenNotMatchedBySource, WriteMode,
};
use lance::index::DatasetIndexExt;
use lance_core::{Error, Result, box_error};
use lance_index::is_system_index;
use lance_index::scalar::FullTextSearchQuery;
use lance_linalg::distance::MetricType;
use lance_namespace::LanceNamespace;
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{
    AlterTableAddColumnsRequest, AlterTableAddColumnsResponse, AlterTableAlterColumnsRequest,
    AlterTableAlterColumnsResponse, AlterTableDropColumnsRequest, AlterTableDropColumnsResponse,
    AlterTransactionRequest, AlterTransactionResponse, AnalyzeTableQueryPlanRequest,
    BatchDeleteTableVersionsRequest, BatchDeleteTableVersionsResponse,
    BranchContents as ModelBranchContents, CountTableRowsRequest, CreateNamespaceRequest,
    CreateNamespaceResponse, CreateTableBranchRequest, CreateTableBranchResponse,
    CreateTableIndexRequest, CreateTableIndexResponse, CreateTableRequest, CreateTableResponse,
    CreateTableScalarIndexResponse, CreateTableTagRequest, CreateTableTagResponse,
    CreateTableVersionRequest, CreateTableVersionResponse, DeclareTableRequest,
    DeclareTableResponse, DeleteFromTableRequest, DeleteFromTableResponse,
    DeleteTableBranchRequest, DeleteTableBranchResponse, DeleteTableTagRequest,
    DeleteTableTagResponse, DescribeNamespaceRequest, DescribeNamespaceResponse,
    DescribeTableIndexStatsRequest, DescribeTableIndexStatsResponse, DescribeTableRequest,
    DescribeTableResponse, DescribeTableVersionRequest, DescribeTableVersionResponse,
    DescribeTransactionRequest, DescribeTransactionResponse, DropNamespaceRequest,
    DropNamespaceResponse, DropTableIndexRequest, DropTableIndexResponse, DropTableRequest,
    DropTableResponse, ExplainTableQueryPlanRequest, FragmentStats, FragmentSummary,
    GetTableStatsRequest, GetTableStatsResponse, GetTableTagVersionRequest,
    GetTableTagVersionResponse, IndexContent, InsertIntoTableRequest, InsertIntoTableResponse,
    ListNamespacesRequest, ListNamespacesResponse, ListTableBranchesRequest,
    ListTableBranchesResponse, ListTableIndicesRequest, ListTableIndicesResponse,
    ListTableTagsRequest, ListTableTagsResponse, ListTableVersionsRequest,
    ListTableVersionsResponse, ListTablesRequest, ListTablesResponse, MergeInsertIntoTableRequest,
    MergeInsertIntoTableResponse, NamespaceExistsRequest, QueryTableRequest, RestoreTableRequest,
    RestoreTableResponse, TableExistsRequest, TagContents as ModelTagContents, UpdateTableRequest,
    UpdateTableResponse, UpdateTableSchemaMetadataRequest, UpdateTableSchemaMetadataResponse,
    UpdateTableTagRequest, UpdateTableTagResponse,
};
use lance_table::io::commit::ManifestNamingScheme;
use object_store::path::Path;
use object_store::{Error as ObjectStoreError, ObjectStoreExt};

use super::marker::{MarkerFileError, put_marker_file_atomic};
use super::query::{apply_probe_bounds, build_engine_fts_query};
use super::versions::{ExistingTableVersionResolve, TableDeleteEntry};
use super::{
    DirectoryNamespace, build_column_alterations, build_sql_expressions, manifest_feature_flags,
};
use crate::merge_insert_on_columns;

#[async_trait]
impl LanceNamespace for DirectoryNamespace {
    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        self.record_op("list_namespaces");
        self.ensure_read_manifest().await?;
        if let Some(manifest_ns) = self.manifest_ns_for_read() {
            return manifest_ns.list_namespaces(request).await;
        }

        if request.id.as_ref().is_some_and(|id| !id.is_empty()) {
            return Err(self.child_namespace_requires_manifest_error());
        }
        Self::validate_root_namespace_id(&request.id)?;
        Ok(ListNamespacesResponse::new(vec![]))
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        self.record_op("describe_namespace");
        self.ensure_read_manifest().await?;
        if let Some(manifest_ns) = self.manifest_ns_for_read() {
            return manifest_ns.describe_namespace(request).await;
        }

        if request.id.as_ref().is_some_and(|id| !id.is_empty()) {
            return Err(self.child_namespace_requires_manifest_error());
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
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
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
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
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
        self.ensure_read_manifest().await?;
        if let Some(manifest_ns) = self.manifest_ns_for_read() {
            return manifest_ns.namespace_exists(request).await;
        }

        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Ok(());
        }

        Err(self.child_namespace_requires_manifest_error())
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        self.record_op("list_tables");
        // Validate that namespace ID is provided
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;

        // Self-heal the manifest wherever it can be authoritative: a child
        // namespace, a manifest-only namespace, or migration mode (which merges
        // manifest entries -- including registered aliases -- into the root
        // listing). The bypass applies only to migration-disabled directory-backed
        // root lists.
        if !namespace_id.is_empty()
            || !self.dir_listing_enabled
            || self.dir_listing_to_manifest_migration_enabled
        {
            self.ensure_read_manifest().await?;
        }

        // For child namespaces, always delegate to manifest (if enabled)
        if !namespace_id.is_empty() {
            if let Some(manifest_ns) = self.manifest_ns_for_read() {
                return manifest_ns.list_tables(request).await;
            }
            return Err(self.child_namespace_requires_manifest_error());
        }

        // When only manifest is enabled (no directory listing), delegate directly to manifest
        if let Some(manifest_ns) = self.manifest_ns_for_read()
            && !self.dir_listing_enabled
        {
            return manifest_ns.list_tables(request).await;
        }
        if !self.dir_listing_enabled {
            return Ok(ListTablesResponse::new(vec![]));
        }

        // When both manifest and directory listing are enabled with migration mode,
        // we need to merge and deduplicate
        let mut tables = if self.manifest_ns_for_read().is_some()
            && self.dir_listing_enabled
            && self.dir_listing_to_manifest_migration_enabled
        {
            // Get all manifest table locations (for deduplication)
            let manifest_locations = if let Some(manifest_ns) = self.manifest_ns_for_read() {
                manifest_ns.list_manifest_table_locations().await?
            } else {
                std::collections::HashSet::new()
            };

            // Get all manifest tables (without pagination for merging)
            let mut manifest_request = request.clone();
            manifest_request.limit = None;
            manifest_request.page_token = None;
            let manifest_tables = if let Some(manifest_ns) = self.manifest_ns_for_read() {
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
        let is_child_table = request.id.as_ref().is_some_and(|id| id.len() > 1);
        let skip_manifest_for_root = self.dir_listing_enabled
            && is_root_level
            && !self.dir_listing_to_manifest_migration_enabled;
        // Child table, manifest-only, or migration mode (see describe_table_impl).
        // Only a migration-disabled directory-backed root read bypasses the probe.
        if is_child_table
            || !self.dir_listing_enabled
            || self.dir_listing_to_manifest_migration_enabled
        {
            self.ensure_read_manifest().await?;
        }
        if let Some(manifest_ns) = self.manifest_ns_for_read()
            && !skip_manifest_for_root
        {
            match manifest_ns.table_exists(request.clone()).await {
                Ok(()) => return Ok(()),
                Err(e) if manifest_feature_flags::is_incompatible_manifest_error(&e) => {
                    // An incompatible manifest must surface "please upgrade"
                    // rather than degrading to a directory-listing view.
                    return Err(e);
                }
                Err(e) if self.dir_listing_enabled && is_root_level => {
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
        if !self.dir_listing_enabled {
            return Err(NamespaceError::TableNotFound { message: table_id }.into());
        }

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

        Ok(())
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        self.record_op("drop_table");
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
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
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
            return manifest_ns.create_table(request, request_data).await;
        }

        Self::validate_dir_only_properties(request.properties.as_ref(), "create_table")?;

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);
        let status = self.check_table_status(&table_name).await?;
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
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
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
        let status = self.check_table_status(&table_name).await?;
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

        put_marker_file_atomic(
            &self.object_store,
            &reserved_file_path,
            &format!("table {}", table_name),
        )
        .await
        .map_err(|e| match e {
            MarkerFileError::AlreadyExists { .. } => {
                lance_core::Error::from(NamespaceError::TableAlreadyExists {
                    message: table_name.to_string(),
                })
            }
            MarkerFileError::Other { message } => {
                lance_core::Error::from(NamespaceError::Internal { message })
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
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
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
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
            return LanceNamespace::deregister_table(manifest_ns.as_ref(), request).await;
        }

        // V1 mode: create a .lance-deregistered marker file in the table directory
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Check table existence and deregistration status.
        // This provides better error messages for common cases.
        let status = self.check_table_status(&table_name).await?;

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
        put_marker_file_atomic(
            &self.object_store,
            &deregistered_path,
            &format!("deregistration marker for table {}", table_name),
        )
        .await
        .map_err(|e| match e {
            MarkerFileError::AlreadyExists { .. } => {
                lance_core::Error::from(NamespaceError::InvalidTableState {
                    message: format!("Table is already deregistered: {}", table_name),
                })
            }
            MarkerFileError::Other { message } => {
                lance_core::Error::from(NamespaceError::Internal { message })
            }
        })?;

        Ok(lance_namespace::models::DeregisterTableResponse {
            id: request.id,
            location: Some(table_uri),
            ..Default::default()
        })
    }

    async fn alter_table_add_columns(
        &self,
        request: AlterTableAddColumnsRequest,
    ) -> Result<AlterTableAddColumnsResponse> {
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
            return manifest_ns.alter_table_add_columns(request).await;
        }

        // Non-manifest mode: open Dataset directly via table URI and perform the operation
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Check table existence and deregistration status before opening the dataset
        let status = self.check_table_status(&table_name).await?;
        if !status.exists {
            return Err(NamespaceError::TableNotFound {
                message: table_name,
            }
            .into());
        }
        if status.is_deregistered {
            return Err(NamespaceError::TableNotFound {
                message: format!("Table is deregistered: {}", table_name),
            }
            .into());
        }

        let mut dataset = self
            .configured_builder(&table_uri)
            .load()
            .await
            .map_err(|e| {
                Error::io_source(box_error(std::io::Error::other(format!(
                    "Failed to open dataset: {}",
                    e
                ))))
            })?;

        let sql_expressions = build_sql_expressions(&request.new_columns)?;

        dataset
            .add_columns(
                lance::dataset::NewColumnTransform::SqlExpressions(sql_expressions),
                None,
                None,
            )
            .await
            .map_err(|e| {
                Error::io_source(box_error(std::io::Error::other(format!(
                    "Failed to add columns: {}",
                    e
                ))))
            })?;

        let version = dataset.version().version as i64;
        Ok(AlterTableAddColumnsResponse::new(version))
    }

    async fn alter_table_alter_columns(
        &self,
        request: AlterTableAlterColumnsRequest,
    ) -> Result<AlterTableAlterColumnsResponse> {
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
            return manifest_ns.alter_table_alter_columns(request).await;
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Check table existence and deregistration status before opening the dataset
        let status = self.check_table_status(&table_name).await?;
        if !status.exists {
            return Err(NamespaceError::TableNotFound {
                message: table_name,
            }
            .into());
        }
        if status.is_deregistered {
            return Err(NamespaceError::TableNotFound {
                message: format!("Table is deregistered: {}", table_name),
            }
            .into());
        }

        let mut dataset = self
            .configured_builder(&table_uri)
            .load()
            .await
            .map_err(|e| {
                Error::io_source(box_error(std::io::Error::other(format!(
                    "Failed to open dataset: {}",
                    e
                ))))
            })?;

        let alterations = build_column_alterations(&request.alterations)?;

        dataset.alter_columns(&alterations).await.map_err(|e| {
            Error::io_source(box_error(std::io::Error::other(format!(
                "Failed to alter columns: {}",
                e
            ))))
        })?;

        let version = dataset.version().version as i64;
        Ok(AlterTableAlterColumnsResponse::new(version))
    }

    async fn alter_table_drop_columns(
        &self,
        request: AlterTableDropColumnsRequest,
    ) -> Result<AlterTableDropColumnsResponse> {
        if let Some(manifest_ns) = self.manifest_ns_for_write().await? {
            return manifest_ns.alter_table_drop_columns(request).await;
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Check table existence and deregistration status before opening the dataset
        let status = self.check_table_status(&table_name).await?;
        if !status.exists {
            return Err(NamespaceError::TableNotFound {
                message: table_name,
            }
            .into());
        }
        if status.is_deregistered {
            return Err(NamespaceError::TableNotFound {
                message: format!("Table is deregistered: {}", table_name),
            }
            .into());
        }

        let mut dataset = self
            .configured_builder(&table_uri)
            .load()
            .await
            .map_err(|e| {
                Error::io_source(box_error(std::io::Error::other(format!(
                    "Failed to open dataset: {}",
                    e
                ))))
            })?;

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        dataset.drop_columns(&columns).await.map_err(|e| {
            Error::io_source(box_error(std::io::Error::other(format!(
                "Failed to drop columns: {}",
                e
            ))))
        })?;

        let version = dataset.version().version as i64;
        Ok(AlterTableDropColumnsResponse::new(version))
    }

    async fn list_table_versions(
        &self,
        request: ListTableVersionsRequest,
    ) -> Result<ListTableVersionsResponse> {
        self.record_op("list_table_versions");
        let branch = Self::normalized_branch(request.branch.as_deref())?;
        let table_uri = self.resolve_table_location(&request.id).await?;
        let table_uri = match branch {
            Some(b) => self.resolve_branch_location(&table_uri, b).await?,
            None => table_uri,
        };
        let want_descending = request.descending == Some(true);
        let table_versions = self
            .list_table_versions_from_storage(&table_uri, want_descending, request.limit)
            .await?;

        Ok(ListTableVersionsResponse {
            versions: table_versions,
            page_token: None,
            ..Default::default()
        })
    }

    async fn create_table_version(
        &self,
        request: CreateTableVersionRequest,
    ) -> Result<CreateTableVersionResponse> {
        self.record_op("create_table_version");
        let branch = Self::normalized_branch(request.branch.as_deref())?;
        let table_uri = self.resolve_table_location(&request.id).await?;
        let (table_uri, table_path, branch_parent_version) = match branch {
            Some(b) => self.resolve_branch_for_commit(&table_uri, b).await?,
            None => {
                let table_path = self.object_store_path_from_uri(&table_uri)?;
                (table_uri, table_path, None)
            }
        };

        let staging_manifest_path = &request.manifest_path;
        let version = request.version as u64;

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

        // Idempotent retry: version path already published with the same content.
        match self.object_store.inner.head(&final_path).await {
            Ok(existing_meta) => {
                return self
                    .resolve_existing_table_version(ExistingTableVersionResolve {
                        staging_path: &staging_path,
                        final_path: &final_path,
                        version,
                        table_uri: &table_uri,
                        final_meta: &existing_meta,
                        request_manifest_size: request.manifest_size,
                    })
                    .await;
            }
            Err(ObjectStoreError::NotFound { .. }) => {}
            Err(e) => {
                return Err(lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to stat version {} for table at '{}': {}",
                        version, table_uri, e
                    ),
                }));
            }
        }

        // Strict CAS: only allow appending latest+1 (or the empty-chain bootstrap
        // version: v1 on main, BranchContents.parent_version on a registered branch).
        let is_branch = branch.is_some();
        self.enforce_create_table_version_cas(
            &table_path,
            version,
            &table_uri,
            is_branch,
            branch_parent_version,
        )
        .await?;

        // Materialize with Create / copy_if_not_exists only — never overwrite.
        let copy_result = self
            .materialize_version_manifest_create(&staging_path, &final_path, staging_manifest_path)
            .await;

        match copy_result {
            Ok(()) => {}
            Err(ObjectStoreError::AlreadyExists { .. })
            | Err(ObjectStoreError::Precondition { .. }) => {
                // Lost a Create race: succeed only if the winner published identical bytes.
                let existing_meta = self.object_store.inner.head(&final_path).await.map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!(
                            "Version {} conflict for table at '{}' but failed to stat winner: {}",
                            version, table_uri, e
                        ),
                    })
                })?;
                return self
                    .resolve_existing_table_version(ExistingTableVersionResolve {
                        staging_path: &staging_path,
                        final_path: &final_path,
                        version,
                        table_uri: &table_uri,
                        final_meta: &existing_meta,
                        request_manifest_size: request.manifest_size,
                    })
                    .await;
            }
            Err(ObjectStoreError::NotFound { .. }) => {
                return Err(lance_core::Error::from(NamespaceError::InvalidInput {
                    message: format!(
                        "Staging manifest not found at '{}' for version {} of table at '{}'",
                        staging_manifest_path, version, table_uri
                    ),
                }));
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

        // Delete the staging manifest after successful copy
        if let Err(e) = self.object_store.inner.delete(&staging_path).await {
            log::warn!(
                "Failed to delete staging manifest at '{}': {:?}",
                staging_path,
                e
            );
        }

        Ok(Self::create_table_version_response(
            version,
            &final_path,
            &final_meta,
        ))
    }

    async fn describe_table_version(
        &self,
        request: DescribeTableVersionRequest,
    ) -> Result<DescribeTableVersionResponse> {
        self.record_op("describe_table_version");
        let branch = Self::normalized_branch(request.branch.as_deref())?;
        let table_uri = self.resolve_table_location(&request.id).await?;
        let table_uri = match branch {
            Some(b) => self.resolve_branch_location(&table_uri, b).await?,
            None => table_uri,
        };
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
            ..Default::default()
        })
    }

    async fn batch_delete_table_versions(
        &self,
        request: BatchDeleteTableVersionsRequest,
    ) -> Result<BatchDeleteTableVersionsResponse> {
        self.record_op("batch_delete_table_versions");
        let branch = Self::normalized_branch(request.branch.as_deref())?;
        // Single-table mode: use `id` (from path parameter) + `ranges` to delete
        // versions from one table.
        let ranges: Vec<(i64, i64)> = request
            .ranges
            .iter()
            .map(|r| (r.start_version, r.end_version))
            .collect();

        // Reject pathological bounded ranges up front: an explicit huge bounded
        // range like (0, i64::MAX) is almost certainly a mistake. A through-latest
        // range (end < 0) is bounded by the manifests that actually exist on storage.
        const MAX_VERSIONS_PER_REQUEST: i128 = 1_000_000;
        let requested: i128 = ranges
            .iter()
            .map(|(s, e)| {
                if *e < 0 {
                    0
                } else {
                    (*e as i128 - *s as i128).max(0)
                }
            })
            .sum();
        if requested > MAX_VERSIONS_PER_REQUEST {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "batch_delete requested {} versions; limit is {}",
                    requested, MAX_VERSIONS_PER_REQUEST
                ),
            }
            .into());
        }

        let table_entries = vec![TableDeleteEntry {
            table_id: request.id.clone(),
            ranges,
        }];

        let total_deleted_count = self
            .delete_physical_version_files(&table_entries, branch)
            .await?;

        Ok(BatchDeleteTableVersionsResponse {
            deleted_count: Some(total_deleted_count),
            transaction_id: None,
            ..Default::default()
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

        Ok(CreateTableIndexResponse {
            transaction_id,
            ..Default::default()
        })
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
        let total_rows = dataset.count_rows(None).await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to count rows for table '{}': {:?}", table_uri, e),
            })
        })? as u64;
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
                            .field_path_minimal(i32::try_from(*field_id).map_err(|e| {
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

                let segments = description.segments();
                let created_at = segments
                    .iter()
                    .filter_map(|segment| segment.created_at)
                    .min()
                    .map(|ts| ts.to_rfc3339());

                // `..Default::default()` keeps this tolerant of additive reqwest
                // client model changes (see #7212).
                #[allow(clippy::needless_update)]
                let content = IndexContent {
                    index_name: description.name().to_string(),
                    index_uuid: description.metadata()[0].uuid.to_string(),
                    columns,
                    status: "SUCCEEDED".to_string(),
                    index_type: Some(description.index_type().to_string()),
                    type_url: Some(description.type_url().to_string()),
                    num_indexed_rows: Some(description.rows_indexed() as i64),
                    num_unindexed_rows: Some(
                        total_rows.saturating_sub(description.rows_indexed()) as i64,
                    ),
                    size_bytes: description.total_size_bytes().map(|size| size as i64),
                    num_segments: Some(segments.len() as i32),
                    created_at,
                    index_version: segments.first().map(|segment| segment.index_version),
                    index_details: description.details().ok(),
                    ..Default::default()
                };
                Ok(content)
            })
            .collect::<Result<Vec<_>>>()?;

        let page_token = Self::paginate_indices(&mut indices, request.page_token, request.limit);
        Ok(ListTableIndicesResponse {
            indexes: indices,
            page_token,
            ..Default::default()
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

        // Merge any persisted alter_transaction changes stored in the sidecar
        // so that describe_transaction reflects the latest altered state.
        let sidecar = self
            .load_transaction_alteration(&table_uri, &transaction.uuid)
            .await?;

        Ok(Self::transaction_response(version, &transaction, sidecar))
    }

    async fn alter_transaction(
        &self,
        request: AlterTransactionRequest,
    ) -> Result<AlterTransactionResponse> {
        self.record_op("alter_transaction");

        // Parse the request ID: must include table id and transaction identifier
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

        let txn_id = request_id.pop().expect("request_id len checked above");
        let table_id = Some(request_id);
        let table_uri = self.resolve_table_location(&table_id).await?;
        let dataset = self
            .load_dataset(&table_uri, None, "alter_transaction")
            .await?;
        let (version, transaction) = self.find_transaction(&dataset, &txn_id).await?;

        // Reserved keys are derived from the immutable Transaction metadata and
        // must not be modified via alter_transaction. They are only surfaced in
        // the response for the caller's convenience.
        const RESERVED_KEYS: &[&str] = &["uuid", "version", "read_version", "operation", "tag"];
        let is_reserved = |key: &str| RESERVED_KEYS.contains(&key);

        // Load the existing sidecar (if any) so alterations accumulate across
        // successive alter_transaction calls.
        let mut sidecar = self
            .load_transaction_alteration(&table_uri, &transaction.uuid)
            .await?
            .unwrap_or_default();

        for action in &request.actions {
            if let Some(ref set_status) = action.set_status_action
                && let Some(ref status) = set_status.status
            {
                // Validate the status value (case-insensitive)
                let normalized = status.to_lowercase().replace('_', "");
                match normalized.as_str() {
                    "queued" | "running" | "succeeded" | "failed" | "canceled" => {
                        sidecar.status = Some(status.clone());
                    }
                    _ => {
                        return Err(NamespaceError::InvalidInput {
                            message: format!(
                                "Invalid transaction status '{}'. Valid values are: Queued, Running, Succeeded, Failed, Canceled",
                                status
                            ),
                        }
                        .into());
                    }
                }
            }

            if let Some(ref set_property) = action.set_property_action
                && let (Some(key), Some(value)) = (&set_property.key, &set_property.value)
            {
                if is_reserved(key) {
                    return Err(NamespaceError::InvalidInput {
                        message: format!("Property '{}' is reserved and cannot be modified", key),
                    }
                    .into());
                }
                let mode = set_property
                    .mode
                    .as_deref()
                    .unwrap_or("Overwrite")
                    .to_lowercase();
                match mode.as_str() {
                    "overwrite" => {
                        sidecar.properties.insert(key.clone(), value.clone());
                    }
                    "fail" => {
                        // Consider both the immutable transaction properties
                        // and any values previously written to the sidecar.
                        let exists = sidecar.properties.contains_key(key)
                            || transaction
                                .transaction_properties
                                .as_ref()
                                .is_some_and(|props| props.contains_key(key));
                        if exists {
                            return Err(NamespaceError::ConcurrentModification {
                                message: format!(
                                    "Property '{}' already exists and mode is 'Fail'",
                                    key
                                ),
                            }
                            .into());
                        }
                        sidecar.properties.insert(key.clone(), value.clone());
                    }
                    "skip" => {
                        let exists = sidecar.properties.contains_key(key)
                            || transaction
                                .transaction_properties
                                .as_ref()
                                .is_some_and(|props| props.contains_key(key));
                        if !exists {
                            sidecar.properties.insert(key.clone(), value.clone());
                        }
                    }
                    _ => {
                        return Err(NamespaceError::InvalidInput {
                            message: format!(
                                "Invalid set_property mode '{}'. Valid values are: Overwrite, Fail, Skip",
                                mode
                            ),
                        }
                        .into());
                    }
                }
            }

            if let Some(ref unset_property) = action.unset_property_action
                && let Some(ref key) = unset_property.key
            {
                if is_reserved(key) {
                    return Err(NamespaceError::InvalidInput {
                        message: format!("Property '{}' is reserved and cannot be modified", key),
                    }
                    .into());
                }
                let mode = unset_property
                    .mode
                    .as_deref()
                    .unwrap_or("Skip")
                    .to_lowercase();
                let exists_in_transaction = transaction
                    .transaction_properties
                    .as_ref()
                    .is_some_and(|props| props.contains_key(key));
                match mode.as_str() {
                    "skip" => {
                        sidecar.properties.remove(key);
                        if exists_in_transaction {
                            // Track a tombstone so describe_transaction can
                            // hide the immutable property from the response.
                            sidecar.removed_properties.insert(key.clone());
                        }
                    }
                    "fail" => {
                        if !sidecar.properties.contains_key(key) && !exists_in_transaction {
                            return Err(NamespaceError::InvalidInput {
                                message: format!(
                                    "Property '{}' does not exist and mode is 'Fail'",
                                    key
                                ),
                            }
                            .into());
                        }
                        sidecar.properties.remove(key);
                        if exists_in_transaction {
                            sidecar.removed_properties.insert(key.clone());
                        }
                    }
                    _ => {
                        return Err(NamespaceError::InvalidInput {
                            message: format!(
                                "Invalid unset_property mode '{}'. Valid values are: Skip, Fail",
                                mode
                            ),
                        }
                        .into());
                    }
                }
            }
        }

        // Persist the accumulated alterations so subsequent calls observe
        // them. The transaction file itself is immutable in Lance, so we
        // record alter_transaction outcomes in a namespace-owned sidecar.
        self.save_transaction_alteration(&table_uri, &transaction.uuid, &sidecar)
            .await?;

        // Assemble the response by merging the immutable transaction metadata
        // with the persisted alterations.
        let final_status = sidecar
            .status
            .clone()
            .unwrap_or_else(|| "SUCCEEDED".to_string());
        let response = Self::transaction_response(version, &transaction, Some(sidecar));
        Ok(AlterTransactionResponse {
            status: final_status,
            properties: response.properties,
            ..Default::default()
        })
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
            ..Default::default()
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

        Ok(DropTableIndexResponse {
            transaction_id,
            ..Default::default()
        })
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

        let branch = Self::normalized_branch(request.branch.as_deref())?;
        let table_uri = self.resolve_table_location(&request.id).await?;
        let mut dataset = match branch {
            Some(branch) => self.open_validated_branch(&table_uri, branch).await?,
            None => self.load_dataset(&table_uri, None, "restore_table").await?,
        };

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

        Ok(RestoreTableResponse {
            transaction_id,
            ..Default::default()
        })
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
            ..Default::default()
        })
    }

    async fn get_table_stats(
        &self,
        request: GetTableStatsRequest,
    ) -> Result<GetTableStatsResponse> {
        let table_uri = self.resolve_table_location(&request.id).await?;
        let branch = request.branch.as_deref();
        let dataset = Arc::new(
            self.load_branch_dataset(&table_uri, branch, None, "get_table_stats")
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
            request.query.minimum_nprobes,
            request.query.maximum_nprobes,
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
            request.minimum_nprobes,
            request.maximum_nprobes,
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
        let branch = request.branch.as_deref();
        let dataset = self
            .load_branch_dataset(&table_uri, branch, request.version, "count_table_rows")
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
        let (reader, num_rows) =
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

        let dataset = if !self.table_uri_has_actual_manifests(&table_uri).await? {
            self.write_reader_to_table(&table_uri, reader, WriteMode::Create, None)
                .await?
        } else {
            self.write_reader_to_table(&table_uri, reader, mode, None)
                .await?
        };

        Ok(InsertIntoTableResponse {
            transaction_id: None,
            num_inserted_rows: Some(num_rows as i64),
            version: Some(dataset.version().version as i64),
            ..Default::default()
        })
    }

    async fn merge_insert_into_table(
        &self,
        request: MergeInsertIntoTableRequest,
        request_data: Bytes,
    ) -> Result<MergeInsertIntoTableResponse> {
        self.record_op("merge_insert_into_table");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let on = merge_insert_on_columns(request.on.as_deref(), "merge_insert_into_table")?;

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
                ..Default::default()
            });
        }

        let dataset = Arc::new(
            self.load_dataset(&table_uri, None, "merge_insert_into_table")
                .await?,
        );

        let mut merge_builder =
            MergeInsertBuilder::try_new(dataset.clone(), on.to_vec()).map_err(|e| {
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
            .map_err(|e| Self::map_mutation_error(e, "merge_insert_into_table", &table_uri))?;

        Ok(MergeInsertIntoTableResponse {
            transaction_id: None,
            num_updated_rows: Some(stats.num_updated_rows as i64),
            num_inserted_rows: Some(stats.num_inserted_rows as i64),
            num_deleted_rows: Some(stats.num_deleted_rows as i64),
            version: Some(dataset.version().version as i64),
            ..Default::default()
        })
    }

    async fn update_table(&self, request: UpdateTableRequest) -> Result<UpdateTableResponse> {
        self.record_op("update_table");

        if request.updates.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "update_table requires at least one [column, expression] pair".to_string(),
            }
            .into());
        }

        // Validate every update pair shape and detect duplicate columns up front so we
        // surface a clean error instead of failing deep inside the planner.
        let mut seen_columns: HashMap<String, ()> = HashMap::with_capacity(request.updates.len());
        for (idx, pair) in request.updates.iter().enumerate() {
            if pair.len() != 2 {
                return Err(NamespaceError::InvalidInput {
                    message: format!(
                        "update_table updates[{}] must be a [column, expression] pair, got {} elements",
                        idx,
                        pair.len()
                    ),
                }
                .into());
            }
            let column = &pair[0];
            if column.trim().is_empty() {
                return Err(NamespaceError::InvalidInput {
                    message: format!("update_table updates[{}] has an empty column name", idx),
                }
                .into());
            }
            if seen_columns.insert(column.clone(), ()).is_some() {
                return Err(NamespaceError::InvalidInput {
                    message: format!(
                        "update_table cannot update column '{}' more than once",
                        column
                    ),
                }
                .into());
            }
        }

        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = Arc::new(self.load_dataset(&table_uri, None, "update_table").await?);

        let mut builder = UpdateBuilder::new(dataset);
        for pair in &request.updates {
            // Indexing by 0/1 is safe due to the length check above.
            builder = builder.set(&pair[0], &pair[1]).map_err(|e| {
                lance_core::Error::from(NamespaceError::InvalidInput {
                    message: format!("Invalid update expression for column '{}': {}", pair[0], e),
                })
            })?;
        }
        if let Some(predicate) = request.predicate.as_deref()
            && !predicate.trim().is_empty()
        {
            builder = builder.update_where(predicate).map_err(|e| {
                lance_core::Error::from(NamespaceError::InvalidInput {
                    message: format!("Invalid update_table predicate '{}': {}", predicate, e),
                })
            })?;
        }

        let job = builder.build().map_err(|e| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: format!("Failed to build update_table job: {}", e),
            })
        })?;

        let result = job
            .execute()
            .await
            .map_err(|e| Self::map_mutation_error(e, "update_table", &table_uri))?;

        let version = result.new_dataset.version().version as i64;
        Ok(UpdateTableResponse {
            transaction_id: None,
            updated_rows: result.rows_updated as i64,
            version,
            properties: None,
            ..Default::default()
        })
    }

    async fn delete_from_table(
        &self,
        request: DeleteFromTableRequest,
    ) -> Result<DeleteFromTableResponse> {
        self.record_op("delete_from_table");

        if request.predicate.trim().is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "delete_from_table requires a non-empty predicate".to_string(),
            }
            .into());
        }

        let table_uri = self.resolve_table_location(&request.id).await?;
        let mut dataset = self
            .load_dataset(&table_uri, None, "delete_from_table")
            .await?;

        let result = dataset
            .delete(&request.predicate)
            .await
            .map_err(|e| Self::map_mutation_error(e, "delete_from_table", &table_uri))?;

        Ok(DeleteFromTableResponse {
            transaction_id: None,
            version: Some(result.new_dataset.version().version as i64),
            ..Default::default()
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

        let query_array =
            Self::query_vector_array(&request.vector, request.offset).map_err(|e| {
                NamespaceError::InvalidInput {
                    message: format!("Invalid vector search for query_table: {}", e),
                }
            })?;
        let has_vector_query = query_array.is_some();

        // Apply prefilter setting (must be set before nearest)
        if let Some(prefilter) = request.prefilter {
            scanner.prefilter(prefilter);
        }

        // Apply vector search if query vector is provided
        if let Some(query_array) = query_array {
            let vector_column = request.vector_column.as_deref().unwrap_or("vector");

            let k = if request.k > 0 {
                request.k as usize
            } else {
                10
            };
            scanner
                .nearest(vector_column, query_array.as_ref(), k)
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

            apply_probe_bounds(
                &mut scanner,
                request.nprobes,
                request.minimum_nprobes,
                request.maximum_nprobes,
            )?;

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
            } else if let Some(ref structured_query) = fts_query.structured_query {
                // Structured FTS: map the namespace query model into the engine FtsQuery.
                let engine_query = build_engine_fts_query(&structured_query.query)?;
                let fts = FullTextSearchQuery::new_query(engine_query);
                scanner
                    .full_text_search(fts)
                    .map_err(|e| NamespaceError::InvalidInput {
                        message: format!("Invalid full text search: {:?}", e),
                    })?;
            }
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
            ..Default::default()
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

        let contents = dataset
            .tags()
            .get(&request.tag)
            .await
            .map_err(|e| Self::map_tag_error(e, &request.tag, &table_uri))?;

        Ok(GetTableTagVersionResponse {
            version: contents.version as i64,
            branch: contents.branch,
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        })
    }

    async fn create_table_branch(
        &self,
        request: CreateTableBranchRequest,
    ) -> Result<CreateTableBranchResponse> {
        self.record_op("create_table_branch");
        if request.name.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "branch name must not be empty for create_table_branch".to_string(),
            }
            .into());
        }
        let from_version = match request.from_version {
            Some(v) if v <= 0 => {
                return Err(NamespaceError::InvalidInput {
                    message: format!(
                        "from_version must be a positive integer, got {} for create_table_branch",
                        v
                    ),
                }
                .into());
            }
            Some(v) => Some(v as u64),
            None => None,
        };

        let table_uri = self.resolve_table_location(&request.id).await?;
        let mut dataset = self
            .load_dataset(&table_uri, None, "create_table_branch")
            .await?;

        // Best-effort pre-check: a duplicate returns a clean TableBranchAlreadyExists conflict
        // instead of the opaque Internal error create_branch raises on a pre-existing branch. A
        // concurrent create can still race past this window. Remove once lance-core create_branch
        // returns RefConflict up front.
        if dataset.branches().get(&request.name).await.is_ok() {
            return Err(NamespaceError::TableBranchAlreadyExists {
                message: format!("branch '{}' for table at '{}'", request.name, table_uri),
            }
            .into());
        }

        dataset
            .create_branch(
                &request.name,
                (request.from_branch.as_deref(), from_version),
                None,
            )
            .await
            .map_err(|e| {
                // After load_dataset + the dup pre-check, a DatasetNotFound from create_branch
                // means the requested fork source (from_branch/from_version) doesn't exist.
                if matches!(e, lance_core::Error::DatasetNotFound { .. }) {
                    NamespaceError::InvalidInput {
                        message: format!(
                            "from_branch/from_version for branch '{}' refers to a source that does not exist: {}",
                            request.name, e
                        ),
                    }
                    .into()
                } else {
                    Self::map_branch_error(e, &request.name, &table_uri)
                }
            })?;

        Ok(CreateTableBranchResponse {
            transaction_id: None,
            ..Default::default()
        })
    }

    async fn list_table_branches(
        &self,
        request: ListTableBranchesRequest,
    ) -> Result<ListTableBranchesResponse> {
        self.record_op("list_table_branches");
        let table_uri = self.resolve_table_location(&request.id).await?;
        let dataset = self
            .load_dataset(&table_uri, None, "list_table_branches")
            .await?;

        let raw_branches = dataset.list_branches().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!(
                    "Failed to list branches for table at '{}': {}",
                    table_uri, e
                ),
            })
        })?;

        let branches = raw_branches
            .into_iter()
            .map(|(name, contents)| {
                // The namespace `BranchContents` model has no `identifier` field, so the
                // lance-core branch identifier is intentionally dropped here.
                let mut branch_model = ModelBranchContents::new(
                    contents.parent_version as i64,
                    contents.create_at as i64,
                    contents.manifest_size as i64,
                );
                branch_model.parent_branch = contents.parent_branch;
                branch_model.metadata = if contents.metadata.is_empty() {
                    None
                } else {
                    Some(contents.metadata)
                };
                (name, branch_model)
            })
            .collect();

        Ok(ListTableBranchesResponse {
            branches,
            page_token: None,
            ..Default::default()
        })
    }

    async fn delete_table_branch(
        &self,
        request: DeleteTableBranchRequest,
    ) -> Result<DeleteTableBranchResponse> {
        self.record_op("delete_table_branch");
        if request.name.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "branch name must not be empty for delete_table_branch".to_string(),
            }
            .into());
        }

        let table_uri = self.resolve_table_location(&request.id).await?;
        let mut dataset = self
            .load_dataset(&table_uri, None, "delete_table_branch")
            .await?;

        dataset
            .delete_branch(&request.name)
            .await
            .map_err(|e| match e {
                lance_core::Error::RefConflict { message } => NamespaceError::InvalidInput {
                    message: format!(
                        "branch '{}' for table at '{}': {}",
                        request.name, table_uri, message
                    ),
                }
                .into(),
                other => Self::map_branch_error(other, &request.name, &table_uri),
            })?;

        Ok(DeleteTableBranchResponse {
            transaction_id: None,
            ..Default::default()
        })
    }

    fn namespace_id(&self) -> String {
        format!("DirectoryNamespace {{ root: {:?} }}", self.root)
    }
}
