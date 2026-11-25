// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use futures::TryStreamExt;
use lance_core::Result;
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{
    CreateTableVersionResponse, ListTableVersionsResponse, TableVersion,
};
use lance_table::io::commit::{ManifestNamingScheme, VERSIONS_DIR};
use object_store::path::Path;
use object_store::{Error as ObjectStoreError, ObjectMeta, ObjectStoreExt, PutMode, PutOptions};

use super::DirectoryNamespace;

fn table_version(version: u64, meta: ObjectMeta) -> TableVersion {
    TableVersion {
        version: version as i64,
        manifest_path: meta.location.to_string(),
        manifest_size: Some(meta.size as i64),
        e_tag: meta.e_tag,
        timestamp_millis: Some(meta.last_modified.timestamp_millis()),
        metadata: None,
    }
}

/// Inputs for resolving an already-published `create_table_version` target.
pub(super) struct ExistingTableVersionResolve<'a> {
    pub(super) staging_path: &'a Path,
    pub(super) final_path: &'a Path,
    pub(super) version: u64,
    pub(super) table_uri: &'a str,
    pub(super) final_meta: &'a ObjectMeta,
    pub(super) request_manifest_size: Option<i64>,
}

/// Describes the version ranges to delete for a single table.
/// Used by `batch_delete_table_versions` and `delete_physical_version_files`.
pub(super) struct TableDeleteEntry {
    pub(super) table_id: Option<Vec<String>>,
    pub(super) ranges: Vec<(i64, i64)>,
}

impl DirectoryNamespace {
    /// Recursive listings also contain staging files, backups and nested manifests.
    pub(super) fn committed_manifest(
        table_path: &Path,
        location: &Path,
    ) -> Option<(ManifestNamingScheme, u64)> {
        let filename = location.filename()?;
        let scheme = ManifestNamingScheme::detect_scheme(filename)?;
        let version = scheme.parse_version(filename)?;
        (scheme.manifest_path(table_path, version) == *location).then_some((scheme, version))
    }

    /// Build a successful `CreateTableVersionResponse` from an existing final manifest.
    pub(super) fn create_table_version_response(
        version: u64,
        final_path: &Path,
        final_meta: &ObjectMeta,
    ) -> CreateTableVersionResponse {
        CreateTableVersionResponse {
            transaction_id: None,
            version: Some(Box::new(TableVersion {
                version: version as i64,
                manifest_path: final_path.to_string(),
                manifest_size: Some(final_meta.size as i64),
                e_tag: final_meta.e_tag.clone(),
                timestamp_millis: None,
                metadata: None,
            })),
            ..Default::default()
        }
    }

    /// Whether the staging blob matches the already-published version blob.
    ///
    /// Used for idempotent retries of `create_table_version`. Object-store
    /// `e_tag` is opaque metadata (not a validated content hash) and may also
    /// change across Create/rename materialize, so it is never used for
    /// identity. Size mismatch is a cheap negative check; byte equality is the
    /// durable success condition.
    async fn staging_matches_final_manifest(
        &self,
        staging_path: &Path,
        final_path: &Path,
        final_meta: &ObjectMeta,
        request_manifest_size: Option<i64>,
    ) -> Result<bool> {
        if let Some(size) = request_manifest_size
            && size != final_meta.size as i64
        {
            return Ok(false);
        }

        let staging_bytes = match self.object_store.read_object(staging_path).await {
            Ok(r) => r.bytes().await.map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to read staging manifest at '{}': {}",
                        staging_path, e
                    ),
                })
            })?,
            Err(ObjectStoreError::NotFound { .. }) => return Ok(false),
            Err(e) => {
                return Err(lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to read staging manifest at '{}': {}",
                        staging_path, e
                    ),
                }));
            }
        };

        let final_bytes = self
            .object_store
            .read_object(final_path)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to read existing version manifest at '{}': {}",
                        final_path, e
                    ),
                })
            })?
            .bytes()
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to read existing version manifest bytes at '{}': {}",
                        final_path, e
                    ),
                })
            })?;

        Ok(staging_bytes.as_ref() == final_bytes.as_ref())
    }

    /// Idempotent success or conflict when the target version path already exists.
    pub(super) async fn resolve_existing_table_version(
        &self,
        args: ExistingTableVersionResolve<'_>,
    ) -> Result<CreateTableVersionResponse> {
        if self
            .staging_matches_final_manifest(
                args.staging_path,
                args.final_path,
                args.final_meta,
                args.request_manifest_size,
            )
            .await?
        {
            // Best-effort cleanup of a retry's staging blob.
            if let Err(e) = self.object_store.inner.delete(args.staging_path).await {
                log::warn!(
                    "Failed to delete staging manifest at '{}': {:?}",
                    args.staging_path,
                    e
                );
            }
            return Ok(Self::create_table_version_response(
                args.version,
                args.final_path,
                args.final_meta,
            ));
        }

        Err(lance_core::Error::from(
            NamespaceError::ConcurrentModification {
                message: format!(
                    "Version {} already exists for table at '{}' with different content",
                    args.version, args.table_uri
                ),
            },
        ))
    }

    /// Enforce version CAS: requested version must be `latest + 1` (or bootstrap).
    ///
    /// Empty-chain bootstrap:
    /// - main must start at v1
    /// - a registered branch must start at `BranchContents.parent_version` (the
    ///   shallow-clone fork version, which may be > 1)
    /// - an unregistered branch (create_branch phase-1, ref not written yet)
    ///   accepts the requested version because `parent_version` is not known yet
    pub(super) async fn enforce_create_table_version_cas(
        &self,
        table_path: &Path,
        version: u64,
        table_uri: &str,
        is_branch: bool,
        branch_parent_version: Option<u64>,
    ) -> Result<()> {
        let latest = self.list_versions_under(table_path, true, Some(1)).await?;
        let expected = match latest.first() {
            Some(v) => (v.version as u64).checked_add(1).ok_or_else(|| {
                lance_core::Error::from(NamespaceError::ConcurrentModification {
                    message: format!(
                        "Version overflow computing next version for table at '{}': \
                             latest version {} cannot advance",
                        table_uri, v.version
                    ),
                })
            })?,
            None => {
                if is_branch {
                    // Prefer BranchContents.parent_version when the ref exists so a
                    // branch forked at v5 cannot bootstrap at an arbitrary version.
                    match branch_parent_version {
                        Some(parent_version) => parent_version,
                        None => version,
                    }
                } else {
                    1
                }
            }
        };
        if version != expected {
            let latest_display = latest
                .first()
                .map(|v| v.version.to_string())
                .unwrap_or_else(|| "none".to_string());
            return Err(lance_core::Error::from(
                NamespaceError::ConcurrentModification {
                    message: format!(
                        "Version CAS failed for table at '{}': requested {}, expected {} (latest {})",
                        table_uri, version, expected, latest_display
                    ),
                },
            ));
        }
        Ok(())
    }

    /// Materialize staging → final with Create semantics only (never overwrite).
    pub(super) async fn materialize_version_manifest_create(
        &self,
        staging_path: &Path,
        final_path: &Path,
        staging_manifest_path: &str,
    ) -> std::result::Result<(), ObjectStoreError> {
        match self
            .object_store
            .copy_if_not_exists(staging_path, final_path)
            .await
        {
            Ok(()) => Ok(()),
            Err(ObjectStoreError::NotImplemented { .. })
            | Err(ObjectStoreError::NotSupported { .. }) => {
                let manifest_data = self
                    .object_store
                    .read_object(staging_path)
                    .await?
                    .bytes()
                    .await
                    .map_err(|e| ObjectStoreError::Generic {
                        store: "DirectoryNamespace",
                        source: Box::new(std::io::Error::other(format!(
                            "Failed to read staging manifest bytes at '{}': {}",
                            staging_manifest_path, e
                        ))),
                    })?;
                self.object_store
                    .put_payload_opts(
                        final_path,
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
        }
    }

    pub(super) async fn list_table_versions_from_storage(
        &self,
        table_uri: &str,
        descending: bool,
        limit: Option<i32>,
    ) -> Result<Vec<TableVersion>> {
        let table_path = self.object_store_path_from_uri(table_uri)?;
        self.list_versions_under(&table_path, descending, limit)
            .await
    }

    /// Page by logical version so cleanup or a manifest naming migration does
    /// not invalidate the last version returned to a client.
    pub(super) async fn list_ascending_version_page(
        &self,
        table_uri: &str,
        limit: Option<i32>,
        page_token: Option<&str>,
    ) -> Result<ListTableVersionsResponse> {
        let after = parse_version_page_token(page_token)?;
        let limit = limit.and_then(|n| usize::try_from(n).ok());
        if limit == Some(0) {
            return Ok(ListTableVersionsResponse::default());
        }
        let table_path = self.object_store_path_from_uri(table_uri)?;
        let versions_dir = table_path.clone().join(VERSIONS_DIR);
        let mut manifests = self.object_store.list_objects(Some(&versions_dir));
        let mut versions = Vec::new();
        let keep = limit.map(|n| n + 1);
        while let Some(meta) =
            manifests
                .try_next()
                .await
                .map_err(|err| NamespaceError::Internal {
                    message: format!("Failed to list manifest files under '{versions_dir}': {err}"),
                })?
        {
            let Some((_, version)) = Self::committed_manifest(&table_path, &meta.location) else {
                continue;
            };
            if after.is_some_and(|after| version <= after as u64) {
                continue;
            }
            versions.push(table_version(version, meta));
            if let Some(keep) = keep
                && versions.len() >= keep.saturating_mul(2)
            {
                // Compact in batches. Storage order need not match numeric order,
                // even on a lexically ordered store with mixed manifest layouts.
                versions.select_nth_unstable_by_key(keep, |v| v.version);
                versions.truncate(keep);
            }
        }
        versions.sort_unstable_by_key(|v| v.version);
        if let Some(keep) = keep {
            versions.truncate(keep);
        }
        if let Some(pair) = versions
            .windows(2)
            .find(|pair| pair[0].version == pair[1].version)
        {
            return Err(NamespaceError::Internal {
                message: format!(
                    "Multiple manifests for version {}: '{}' and '{}'",
                    pair[0].version, pair[0].manifest_path, pair[1].manifest_path
                ),
            }
            .into());
        }
        let page_token = if let Some(limit) = limit
            && versions.len() > limit
        {
            versions.truncate(limit);
            versions.last().map(|last| last.version.to_string())
        } else {
            None
        };
        Ok(ListTableVersionsResponse {
            versions,
            page_token,
            ..Default::default()
        })
    }

    /// Newest-first counterpart of [`Self::list_ascending_version_page`]. The
    /// token is the last (oldest) version returned, and the next page holds
    /// the versions below it. Ordered V2 stores keep the early stop, so the
    /// `get_latest_version` hot path still reads only `limit + 1` manifests.
    pub(super) async fn list_descending_version_page(
        &self,
        table_uri: &str,
        limit: Option<i32>,
        page_token: Option<&str>,
    ) -> Result<ListTableVersionsResponse> {
        let before = parse_version_page_token(page_token)?;
        let limit = limit.and_then(|n| usize::try_from(n).ok());
        if limit == Some(0) {
            return Ok(ListTableVersionsResponse::default());
        }
        let table_path = self.object_store_path_from_uri(table_uri)?;
        // One extra row tells whether another page follows.
        let mut versions = self
            .list_versions_below(
                &table_path,
                true,
                limit.map(|n| n.saturating_add(1)),
                before.map(|v| v as u64),
            )
            .await?;
        let page_token = if let Some(limit) = limit
            && versions.len() > limit
        {
            versions.truncate(limit);
            versions.last().map(|last| last.version.to_string())
        } else {
            None
        };
        Ok(ListTableVersionsResponse {
            versions,
            page_token,
            ..Default::default()
        })
    }

    /// List committed manifest versions under `table_path/_versions/`.
    /// `table_path` must be an object-store `Path`; converting a URI to a path
    /// can miss manifests on Windows.
    pub(super) async fn list_versions_under(
        &self,
        table_path: &Path,
        descending: bool,
        limit: Option<i32>,
    ) -> Result<Vec<TableVersion>> {
        let limit = limit.and_then(|limit| usize::try_from(limit).ok());
        self.list_versions_below(table_path, descending, limit, None)
            .await
    }

    /// [`Self::list_versions_under`], keeping only versions below `before`.
    pub(super) async fn list_versions_below(
        &self,
        table_path: &Path,
        descending: bool,
        limit: Option<usize>,
        before: Option<u64>,
    ) -> Result<Vec<TableVersion>> {
        if limit == Some(0) {
            return Ok(Vec::new());
        }
        let versions_dir = table_path.clone().join(VERSIONS_DIR);
        let mut stream = self
            .object_store
            .read_dir_all(&versions_dir, None)
            .try_filter_map(|meta| {
                let manifest = Self::committed_manifest(table_path, &meta.location)
                    .map(|(scheme, version)| (scheme, table_version(version, meta)));
                std::future::ready(Ok(manifest))
            });
        let list_err = |e: lance_core::Error| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!(
                    "Failed to list manifest files under '{}': {}",
                    versions_dir, e
                ),
            })
        };

        let Some((scheme, first)) = stream.try_next().await.map_err(list_err)? else {
            return Ok(Vec::new());
        };
        // Bounded after the scheme is read: the order check below depends on
        // the first manifest listed, not the first one kept.
        let in_range = |v: &TableVersion| before.is_none_or(|before| (v.version as u64) < before);
        let mut table_versions = Vec::new();
        if in_range(&first) {
            table_versions.push(first);
        }

        // Only V2 names may be trusted to match stream order: V2 filenames
        // invert and zero-pad the version, so a lexically-ordered stream
        // arrives newest-first and a descending request can stop after
        // `limit` manifests instead of paginating the whole directory (the
        // `get_latest_version` hot path: descending, limit 1). V1 names are
        // unpadded, so lexical order diverges from version order across
        // digit widths (`10.manifest` sorts before `2.manifest`); V1
        // listings are always collected and sorted numerically.
        let list_is_ordered = self.object_store.list_is_lexically_ordered;
        let stream_matches_request =
            list_is_ordered && scheme == ManifestNamingScheme::V2 && descending;
        //
        // A V2 first entry is taken to mean the table uses V2 names, the same
        // assumption `resolve_version_from_listing` makes when a dataset is
        // opened, so the latest-version lookup stays one bounded listing on
        // every backend. A V1 manifest met while scanning (a partly migrated
        // table, or one sorting between V2 names) drops the early stop, and
        // the listing is collected and sorted numerically. V1 names that sort
        // after every manifest the early stop reads are not seen; such a table
        // should finish `migrate_manifest_paths_v2`.
        let mut early_stop_at = limit.filter(|_| stream_matches_request);

        while early_stop_at.is_none_or(|n| table_versions.len() < n) {
            match stream.try_next().await.map_err(list_err)? {
                Some((manifest_scheme, manifest)) => {
                    if manifest_scheme != scheme {
                        early_stop_at = None;
                    }
                    if in_range(&manifest) {
                        table_versions.push(manifest);
                    }
                }
                None => break,
            }
        }

        if early_stop_at.is_none() {
            if descending {
                table_versions.sort_by_key(|v| std::cmp::Reverse(v.version));
            } else {
                table_versions.sort_by_key(|v| v.version);
            }
            if let Some(limit) = limit {
                table_versions.truncate(limit);
            }
        }

        Ok(table_versions)
    }

    /// Delete physical manifest files for the given table version ranges.
    ///
    /// This helper backs `batch_delete_table_versions`. It resolves each table's storage
    /// location, computes the version file paths, and deletes them, returning an error on
    /// the first failure.
    ///
    /// Returns the number of files successfully deleted.
    pub(super) async fn delete_physical_version_files(
        &self,
        table_entries: &[TableDeleteEntry],
        branch: Option<&str>,
    ) -> Result<i64> {
        let mut deleted_count = 0i64;
        for te in table_entries {
            let table_uri = self.resolve_table_location(&te.table_id).await?;
            let table_uri = match branch {
                Some(b) => self.resolve_branch_location(&table_uri, b).await?,
                None => table_uri,
            };
            let table_path = self.object_store_path_from_uri(&table_uri)?;
            let versions_dir_path = table_path.clone().join(VERSIONS_DIR);

            // Match listed files, not constructed names (`{version}.manifest` misses V2).
            let manifest_metas: Vec<_> = self
                .object_store
                .read_dir_all(&versions_dir_path, None)
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
            let location_by_version: HashMap<u64, Path> = manifest_metas
                .into_iter()
                .filter_map(|meta| {
                    let (_, version) = Self::committed_manifest(&table_path, &meta.location)?;
                    Some((version, meta.location))
                })
                .collect();

            for (&v, version_path) in &location_by_version {
                let vi = v as i64;
                if !te.ranges.iter().any(|&(s, e)| vi >= s && (e < 0 || vi < e)) {
                    continue;
                }
                match self.object_store.inner.delete(version_path).await {
                    Ok(_) => {
                        deleted_count += 1;
                    }
                    Err(object_store::Error::NotFound { .. }) => {}
                    Err(e) => {
                        return Err(NamespaceError::Internal {
                            message: format!(
                                "Failed to delete version {} for table at '{}': {}",
                                v, table_uri, e
                            ),
                        }
                        .into());
                    }
                }
            }
        }
        Ok(deleted_count)
    }
}

/// Parse a table version `page_token`: the canonical decimal form of a
/// non-negative version. Empty or absent means the first page.
fn parse_version_page_token(page_token: Option<&str>) -> Result<Option<i64>> {
    Ok(page_token
        .filter(|token| !token.is_empty())
        .map(|token| {
            token
                .parse::<i64>()
                .ok()
                .filter(|v| *v >= 0 && v.to_string() == token)
                .ok_or_else(|| NamespaceError::InvalidInput {
                    message: format!("Invalid table version page_token '{token}'"),
                })
        })
        .transpose()?)
}
