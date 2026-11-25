// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance::dataset::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::refs::{BRANCH_DIR, BranchIdentifier, BranchLocation, Refs, check_valid_tag};
use lance_core::{Error, Result};
use lance_namespace::LanceNamespace;
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{DescribeTableRequest, DescribeTableResponse};
use lance_namespace::schema::arrow_schema_to_json;

use super::{DirectoryNamespace, manifest, manifest_feature_flags};

enum SnapshotCache<'a> {
    Default,
    Isolated,
    Branch(&'a BranchIdentifier),
}

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

        if let Some(tag) = request.tag.as_deref()
            && (request.branch.is_some() || request.version.is_some())
        {
            return Err(NamespaceError::InvalidInput {
                message: format!("Tag '{tag}' cannot be combined with a version or branch"),
            }
            .into());
        }
        let branch = Self::normalized_branch(request.branch.as_deref())?;
        let version = request
            .version
            .map(|version| {
                u64::try_from(version).map_err(|_| NamespaceError::InvalidInput {
                    message: format!("Table version must be non-negative, got {version}"),
                })
            })
            .transpose()?;
        if load_detailed_metadata && is_only_declared == Some(true) {
            if let Some(tag) = request.tag.as_deref() {
                check_valid_tag(tag).map_err(|err| Self::map_tag_error(err, tag, &table_uri))?;
                return Err(NamespaceError::TableTagNotFound {
                    message: format!("Tag '{tag}' does not exist on declared table '{table_name}'"),
                }
                .into());
            }
            if let Some(branch) = branch {
                return Err(NamespaceError::TableBranchNotFound {
                    message: format!(
                        "Branch '{branch}' does not exist on declared table '{table_name}'"
                    ),
                }
                .into());
            }
            if let Some(version) = version {
                return Err(NamespaceError::TableVersionNotFound {
                    message: format!(
                        "Version {version} does not exist on declared table '{table_name}'"
                    ),
                }
                .into());
            }
        }
        if !load_detailed_metadata || is_only_declared == Some(true) {
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

        let dataset = match (request.tag.as_deref(), branch) {
            (Some(tag), _) => Box::pin(self.open_directory_tag(&table_uri, tag)).await?,
            (None, Some(name)) => {
                // Keep ref resolution out of the future used by shallow location lookups.
                Box::pin(self.open_directory_branch(&table_uri, name, version)).await?
            }
            (None, None) => {
                let dataset = self
                    .open_directory_version(&table_uri, version, SnapshotCache::Default)
                    .await?;
                match dataset.manifest().branch.as_deref() {
                    Some(_) if request.branch.is_some() => {
                        return Err(NamespaceError::Unsupported {
                            message: format!(
                                "The main branch is outside the directory-listed table '{table_uri}'"
                            ),
                        }
                        .into());
                    }
                    Some(name) => {
                        let expected = dataset
                            .branch_identifier()
                            .await
                            .map_err(|err| Self::describe_branch_error(err, name, &table_uri))?;
                        self.open_verified_directory_branch(&table_uri, name, version, &expected)
                            .await?
                    }
                    None => dataset,
                }
            }
        };
        let version_info = dataset.version();
        let wire_version =
            i64::try_from(version_info.version).map_err(|_| NamespaceError::Unsupported {
                message: format!(
                    "Version {} exceeds the signed 64-bit namespace version range",
                    version_info.version
                ),
            })?;
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
            version: Some(wire_version),
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

    async fn open_directory_version(
        &self,
        uri: &str,
        version: Option<u64>,
        cache: SnapshotCache<'_>,
    ) -> Result<Dataset> {
        let mut builder = self.configured_builder(uri);
        if let Some(session) = &self.session
            && (!matches!(cache, SnapshotCache::Default)
                || self
                    .base_path
                    .parts()
                    .any(|part| part.as_ref() == BRANCH_DIR))
        {
            // An ordinary table and a branch can successively occupy the same
            // path below a tree directory. Main tags and legacy branch refs
            // also have no generation UUID to distinguish earlier occupants.
            let generation = match cache {
                SnapshotCache::Branch(id) => id
                    .version_mapping
                    .last()
                    .filter(|(version, _)| *version > 0)
                    .map(|(_, uuid)| uuid.as_str()),
                SnapshotCache::Default | SnapshotCache::Isolated => None,
            };
            builder = builder.with_session(session.with_cache_identity(generation).into());
        }
        let builder = match version {
            Some(version) => builder.with_version(version),
            None => builder,
        };
        builder.load().await.map_err(|err| {
            let message = format!("Failed to open table '{uri}' with version {version:?}: {err}");
            Self::describe_version_error(err, message)
        })
    }

    async fn directory_refs(&self, table_uri: &str) -> Result<(BranchLocation, Refs)> {
        let (store, path, handler) = self
            .configured_builder(table_uri)
            .build_object_store()
            .await?;
        let root = BranchLocation {
            uri: table_uri.to_owned(),
            path,
            branch: None,
        };
        let refs = Refs::new(store, handler, root.clone());
        Ok((root, refs))
    }

    async fn open_directory_tag(&self, table_uri: &str, tag: &str) -> Result<Dataset> {
        let (root, refs) = self.directory_refs(table_uri).await?;
        let tag_error = |err| match err {
            Error::IO { .. } => Self::classify_storage_error(err),
            _ => Self::map_tag_error(err, tag, table_uri),
        };
        let target = match refs.tags().get(tag).await {
            Ok(target) => target,
            Err(missing @ Error::RefNotFound { .. })
                if self
                    .base_path
                    .parts()
                    .any(|part| part.as_ref() == BRANCH_DIR) =>
            {
                // A directly listed branch stores its tags at the owning table's root.
                // Establish ownership from its manifest before looking outside the listing.
                let listed = self
                    .open_directory_version(table_uri, None, SnapshotCache::Isolated)
                    .await?;
                let Some(owner) = listed.manifest().branch.as_deref() else {
                    return Err(tag_error(missing));
                };
                let expected = listed
                    .branch_identifier()
                    .await
                    .map_err(|err| Self::describe_branch_error(err, owner, table_uri))?;
                let target = listed.tags().get(tag).await.map_err(tag_error)?;
                if target.branch.as_deref() != Some(owner) {
                    return Err(NamespaceError::Unsupported {
                        message: format!(
                            "Tag '{tag}' points outside the directory-listed branch '{owner}'"
                        ),
                    }
                    .into());
                }
                return self
                    .open_verified_directory_branch(
                        table_uri,
                        owner,
                        Some(target.version),
                        &expected,
                    )
                    .await;
            }
            Err(err) => return Err(tag_error(err)),
        };
        let confirm_target = async || {
            let current = refs.tags().get(tag).await.map_err(tag_error)?;
            if current.branch != target.branch
                || current.version != target.version
                || current.created_at != target.created_at
            {
                return Err(Error::from(NamespaceError::ConcurrentModification {
                    message: format!("Tag '{tag}' changed while describing '{table_uri}'"),
                }));
            }
            Ok(())
        };
        if let Some(branch) = Self::normalized_branch(target.branch.as_deref())? {
            let expected = refs
                .branches()
                .get_identifier(Some(branch))
                .await
                .map_err(|err| Self::describe_branch_error(err, branch, table_uri))?;
            // A tag stores a branch name and version, without a generation.
            // Confirm the target while the captured branch generation is live.
            confirm_target().await?;
            let location = root.find_branch(Some(branch))?;
            return self
                .open_verified_directory_branch(
                    &location.uri,
                    branch,
                    Some(target.version),
                    &expected,
                )
                .await;
        }
        let snapshot = self
            .open_directory_version(table_uri, Some(target.version), SnapshotCache::Isolated)
            .await?;
        if snapshot.manifest().branch.is_some() {
            return Err(NamespaceError::Unsupported {
                message: format!(
                    "Tag '{tag}' refers to main outside the directory-listed table '{table_uri}'"
                ),
            }
            .into());
        }
        // Dropping and recreating a table can reuse the same main manifest URI.
        confirm_target().await?;
        Ok(snapshot)
    }

    /// Resolve the branch ref before opening its snapshot. The default branch's
    /// latest manifest may require reader features that the selected branch does not.
    async fn open_directory_branch(
        &self,
        table_uri: &str,
        branch: &str,
        version: Option<u64>,
    ) -> Result<Dataset> {
        let (root, refs) = self.directory_refs(table_uri).await?;
        let map_error = |err| Self::describe_branch_error(err, branch, table_uri);
        let (location, expected) = match refs.branches().get_identifier(Some(branch)).await {
            Ok(identifier) => (root.find_branch(Some(branch))?, identifier),
            Err(err @ Error::RefNotFound { .. }) => {
                // A table listed directly beneath a tree directory may itself
                // be a branch. Establish its owner from the selected snapshot
                // before using that owner's identifier to validate the read.
                let listed = BranchLocation {
                    branch: Some(branch.to_owned()),
                    ..root
                };
                if listed.find_main().is_err() {
                    return Err(map_error(err));
                }
                let selected = self
                    .open_directory_version(table_uri, version, SnapshotCache::Isolated)
                    .await?;
                if selected.manifest().branch.as_deref() != Some(branch) {
                    return Err(map_error(err));
                }
                let identifier = selected.branch_identifier().await.map_err(map_error)?;
                (listed, identifier)
            }
            Err(err) => return Err(map_error(err)),
        };
        self.open_verified_directory_branch(&location.uri, branch, version, &expected)
            .await
    }

    async fn open_verified_directory_branch(
        &self,
        uri: &str,
        branch: &str,
        version: Option<u64>,
        expected: &BranchIdentifier,
    ) -> Result<Dataset> {
        let dataset = self
            .open_directory_version(uri, version, SnapshotCache::Branch(expected))
            .await?;
        let actual = dataset
            .branch_identifier()
            .await
            .map_err(|err| Self::describe_branch_error(err, branch, uri))?;
        if dataset.manifest().branch.as_deref() != Some(branch) || &actual != expected {
            return Err(NamespaceError::ConcurrentModification {
                message: format!("Branch '{branch}' changed while describing '{uri}'"),
            }
            .into());
        }
        Ok(dataset)
    }

    fn describe_branch_error(err: Error, branch: &str, uri: &str) -> Error {
        match err {
            Error::IO { .. } => Self::classify_storage_error(err),
            _ => Self::map_branch_error(err, branch, uri),
        }
    }

    fn describe_version_error(err: Error, message: String) -> Error {
        match err.find_root() {
            Error::RefNotFound { .. } => NamespaceError::TableBranchNotFound { message }.into(),
            Error::VersionNotFound { .. } | Error::DatasetNotFound { .. } => {
                NamespaceError::TableVersionNotFound { message }.into()
            }
            Error::NotSupported { .. } => NamespaceError::Unsupported { message }.into(),
            _ if manifest::ManifestNamespace::is_not_found_load_error(&err) => {
                NamespaceError::TableVersionNotFound { message }.into()
            }
            _ => Self::classify_storage_error(err),
        }
    }

    /// Build a `DatasetBuilder` for `table_uri` with this namespace's storage
    /// options and session applied. Callers add version/branch scoping.
    pub(super) fn configured_builder(&self, table_uri: &str) -> DatasetBuilder {
        let mut builder = DatasetBuilder::from_uri(table_uri);
        if let Some(opts) = &self.storage_options {
            builder = builder.with_storage_options(opts.clone());
        }
        if let Some(sess) = &self.session {
            builder = builder.with_session(sess.clone());
        }
        builder
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
