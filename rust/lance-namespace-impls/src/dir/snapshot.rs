// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance::dataset::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::refs::{
    BRANCH_DIR, BranchIdentifier, BranchLocation, Refs, TagContents, check_valid_tag,
};
use lance::session::Session;
use lance_core::{Error, Result};
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{DescribeTableRequest, DescribeTableResponse};
use lance_namespace::schema::arrow_schema_to_json;
use object_store::path::Path;
use std::{collections::HashMap, sync::Arc};

use super::{DirectoryNamespace, manifest};

enum SnapshotCache<'a> {
    Default,
    Isolated,
    Branch(&'a BranchIdentifier),
}

/// Opens a snapshot at the physical location established by the namespace.
pub(super) struct SnapshotReader<'a> {
    pub(super) table_path: &'a Path,
    pub(super) storage_options: Option<&'a HashMap<String, String>>,
    pub(super) session: Option<&'a Arc<Session>>,
}

impl SnapshotReader<'_> {
    pub(super) async fn describe(
        &self,
        request: &DescribeTableRequest,
        table_name: &str,
        table_uri: &str,
        is_only_declared: Option<bool>,
    ) -> Result<DescribeTableResponse> {
        let load_detailed_metadata = request.load_detailed_metadata.unwrap_or(false);
        if let Some(tag) = request.tag.as_deref()
            && (request.branch.is_some() || request.version.is_some())
        {
            return Err(NamespaceError::InvalidInput {
                message: format!("Tag '{tag}' cannot be combined with a version or branch"),
            }
            .into());
        }
        let branch = DirectoryNamespace::normalized_branch(request.branch.as_deref())?;
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
                check_valid_tag(tag)
                    .map_err(|err| DirectoryNamespace::map_tag_error(err, tag, table_uri))?;
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
            return Ok(DescribeTableResponse::default());
        }

        // Shallow location lookups do not need a snapshot reader future.
        Box::pin(self.describe_snapshot(table_uri, request, branch, version)).await
    }

    async fn describe_snapshot(
        &self,
        table_uri: &str,
        request: &DescribeTableRequest,
        branch: Option<&str>,
        version: Option<u64>,
    ) -> Result<DescribeTableResponse> {
        let dataset = match (request.tag.as_deref(), branch) {
            (Some(tag), _) => self.open_tag(table_uri, tag).await?,
            (None, Some(name)) => self.open_branch(table_uri, name, version).await?,
            (None, None) => {
                let dataset = self
                    .open_version(table_uri, version, SnapshotCache::Default)
                    .await?;
                match dataset.manifest().branch.as_deref() {
                    Some(_) if request.branch.is_some() => {
                        return Err(NamespaceError::Unsupported {
                            message: format!("The main branch is outside the table '{table_uri}'"),
                        }
                        .into());
                    }
                    Some(name) => {
                        let expected = dataset
                            .branch_identifier()
                            .await
                            .map_err(|err| Self::describe_branch_error(err, name, table_uri))?;
                        self.open_verified_branch(table_uri, name, version, &expected)
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
        Ok(DescribeTableResponse {
            version: Some(wire_version),
            schema: Some(Box::new(json_schema)),
            metadata: Some(version_info.metadata.into_iter().collect()),
            ..Default::default()
        })
    }

    async fn open_version(
        &self,
        uri: &str,
        version: Option<u64>,
        cache: SnapshotCache<'_>,
    ) -> Result<Dataset> {
        let mut builder = self.configured_builder(uri);
        if let Some(session) = self.session
            && (!matches!(cache, SnapshotCache::Default)
                || self
                    .table_path
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

    async fn table_refs(&self, table_uri: &str) -> Result<(BranchLocation, Refs)> {
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

    async fn open_tag(&self, table_uri: &str, tag: &str) -> Result<Dataset> {
        let (root, refs) = self.table_refs(table_uri).await?;
        let tag_error = |err| match err {
            Error::IO { .. } => DirectoryNamespace::classify_storage_error(err),
            _ => DirectoryNamespace::map_tag_error(err, tag, table_uri),
        };
        // The namespace can prune and republish a version within a live branch.
        // Confirm the tag through the opened snapshot's owner after reading the manifest.
        let confirm_target = async |snapshot: &Dataset, target: &TagContents| {
            let current = snapshot.tags().get(tag).await.map_err(tag_error)?;
            if current.branch != target.branch
                || current.version != target.version
                || current.created_at != target.created_at
                || current.updated_at != target.updated_at
            {
                return Err(Error::from(NamespaceError::ConcurrentModification {
                    message: format!("Tag '{tag}' changed while describing '{table_uri}'"),
                }));
            }
            Ok(())
        };
        let target = match refs.tags().get(tag).await {
            Ok(target) => target,
            Err(missing @ Error::RefNotFound { .. })
                if self
                    .table_path
                    .parts()
                    .any(|part| part.as_ref() == BRANCH_DIR) =>
            {
                // A directly listed branch stores its tags at the owning table's root.
                // Establish ownership from its manifest before looking outside the listing.
                let listed = self
                    .open_version(table_uri, None, SnapshotCache::Isolated)
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
                        message: format!("Tag '{tag}' points outside the branch '{owner}'"),
                    }
                    .into());
                }
                let snapshot = self
                    .open_verified_branch(table_uri, owner, Some(target.version), &expected)
                    .await?;
                confirm_target(&snapshot, &target).await?;
                return Ok(snapshot);
            }
            Err(err) => return Err(tag_error(err)),
        };
        if let Some(branch) = DirectoryNamespace::normalized_branch(target.branch.as_deref())? {
            let expected = refs
                .branches()
                .get_identifier(Some(branch))
                .await
                .map_err(|err| Self::describe_branch_error(err, branch, table_uri))?;
            let location = root.find_branch(Some(branch))?;
            let snapshot = self
                .open_verified_branch(&location.uri, branch, Some(target.version), &expected)
                .await?;
            confirm_target(&snapshot, &target).await?;
            return Ok(snapshot);
        }
        let snapshot = self
            .open_version(table_uri, Some(target.version), SnapshotCache::Isolated)
            .await?;
        if snapshot.manifest().branch.is_some() {
            return Err(NamespaceError::Unsupported {
                message: format!("Tag '{tag}' refers to main outside the table '{table_uri}'"),
            }
            .into());
        }
        // Restoring a table can bring back the original tag, including its timestamp.
        // Compare the manifests on both sides of the confirming tag read.
        confirm_target(&snapshot, &target).await?;
        let current = self
            .open_version(table_uri, Some(target.version), SnapshotCache::Isolated)
            .await?;
        if current.manifest() != snapshot.manifest() {
            return Err(NamespaceError::ConcurrentModification {
                message: format!(
                    "Tagged manifest changed while describing '{tag}' at '{table_uri}'"
                ),
            }
            .into());
        }
        Ok(snapshot)
    }

    /// Resolve the branch ref before opening its snapshot. The default branch's
    /// latest manifest may require reader features that the selected branch does not.
    async fn open_branch(
        &self,
        table_uri: &str,
        branch: &str,
        version: Option<u64>,
    ) -> Result<Dataset> {
        let (root, refs) = self.table_refs(table_uri).await?;
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
                    .open_version(table_uri, version, SnapshotCache::Isolated)
                    .await?;
                if selected.manifest().branch.as_deref() != Some(branch) {
                    return Err(map_error(err));
                }
                let identifier = selected.branch_identifier().await.map_err(map_error)?;
                (listed, identifier)
            }
            Err(err) => return Err(map_error(err)),
        };
        self.open_verified_branch(&location.uri, branch, version, &expected)
            .await
    }

    async fn open_verified_branch(
        &self,
        uri: &str,
        branch: &str,
        version: Option<u64>,
        expected: &BranchIdentifier,
    ) -> Result<Dataset> {
        let dataset = self
            .open_version(uri, version, SnapshotCache::Branch(expected))
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
            Error::IO { .. } => DirectoryNamespace::classify_storage_error(err),
            _ => DirectoryNamespace::map_branch_error(err, branch, uri),
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
            _ => DirectoryNamespace::classify_storage_error(err),
        }
    }

    pub(super) fn configured_builder(&self, table_uri: &str) -> DatasetBuilder {
        let mut builder = DatasetBuilder::from_uri(table_uri);
        if let Some(opts) = self.storage_options {
            builder = builder.with_storage_options(opts.clone());
        }
        if let Some(sess) = self.session {
            builder = builder.with_session(sess.clone());
        }
        builder
    }
}
