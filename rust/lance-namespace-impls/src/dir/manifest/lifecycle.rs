// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::time::Duration;

use lance::index::DatasetIndexExt;
use lance_core::Result;
use lance_namespace::error::NamespaceError;
use lance_table::feature_flags::ensure_can_write_manifest;
use lance_table::format::Manifest;
use lance_table::io::commit::CommitError;
use lance_table::transaction::{Operation, Transaction, translate_config_updates};

use super::ManifestNamespace;
use crate::dir::manifest_feature_flags::{
    NAMESPACE_LIFECYCLE, READER_FEATURE_FLAGS_KEY, WRITER_FEATURE_FLAGS_KEY, ensure_writable,
    reader_flags, writer_flags,
};

/// Committed metadata marker identifying a permanently deleted namespace root.
pub const DELETED_METADATA_KEY: &str = "lance.namespace.manifest.deleted";

fn is_deleted(metadata: &HashMap<String, String>) -> Result<bool> {
    match metadata.get(DELETED_METADATA_KEY).map(String::as_str) {
        None | Some("false") => Ok(false),
        Some("true") => Ok(true),
        Some(value) => Err(NamespaceError::Unsupported {
            message: format!("Invalid namespace deleted marker '{value}'"),
        }
        .into()),
    }
}

pub(super) fn ensure_live(metadata: &HashMap<String, String>) -> Result<()> {
    if is_deleted(metadata)? {
        return Err(NamespaceError::NamespaceNotFound {
            message: "The namespace root has been deleted".into(),
        }
        .into());
    }
    Ok(())
}

impl ManifestNamespace {
    pub(crate) async fn ensure_live(&self) -> Result<()> {
        let dataset = self.manifest_dataset.get().await?;
        ensure_live(dataset.metadata())
    }

    pub(crate) async fn is_deleted(&self) -> Result<bool> {
        let dataset = self.manifest_dataset.get_for_lifecycle().await?;
        is_deleted(dataset.metadata())
    }

    pub(crate) async fn drop_root_namespace(&self) -> Result<bool> {
        let _guard = self.manifest_mutation_lock.lock().await;
        let handler = self.manifest_commit_handler().await?;
        let max_retries = self.manifest_rewrite_commit_retries();
        let mut attempt = 0;
        loop {
            let dataset = self.manifest_dataset.get_for_lifecycle().await?.clone();
            if is_deleted(dataset.metadata())? {
                return Ok(false);
            }
            ensure_can_write_manifest(dataset.manifest())?;
            ensure_writable(dataset.metadata())?;
            let mut scanner = dataset.scan();
            scanner.filter("object_type = 'table'")?;
            if scanner.count_rows().await? != 0 {
                return Err(NamespaceError::NamespaceNotEmpty {
                    message: format!("Namespace '{}' contains table registrations", self.root),
                }
                .into());
            }

            let updates = HashMap::from([
                (DELETED_METADATA_KEY.to_string(), "true".to_string()),
                (
                    READER_FEATURE_FLAGS_KEY.to_string(),
                    (reader_flags(dataset.metadata())? | NAMESPACE_LIFECYCLE).to_string(),
                ),
                (
                    WRITER_FEATURE_FLAGS_KEY.to_string(),
                    (writer_flags(dataset.metadata())? | NAMESPACE_LIFECYCLE).to_string(),
                ),
            ]);
            let transaction = Transaction::new_from_version(
                dataset.version().version,
                Operation::UpdateConfig {
                    config_updates: None,
                    table_metadata_updates: Some(translate_config_updates(&updates, &[])),
                    schema_metadata_updates: None,
                    field_metadata_updates: HashMap::new(),
                },
            );
            let previous = dataset.manifest();
            let mut manifest = Manifest::new_from_previous(
                previous,
                previous.schema.clone(),
                previous.fragments.clone(),
            );
            manifest.table_metadata.extend(updates);
            let indices = dataset.load_indices().await?;
            // Commit exactly the snapshot whose emptiness we checked; never rebase the marker.
            let result = self
                .commit_manifest_overwrite(
                    &dataset,
                    handler.as_ref(),
                    &mut manifest,
                    Some(indices.as_ref().clone()),
                    transaction,
                )
                .await;
            match result {
                Ok(()) => return Ok(true),
                Err(err) => {
                    if let Ok(committed) = dataset.checkout_version(manifest.version).await
                        && is_deleted(committed.metadata())?
                    {
                        return Ok(true);
                    }
                    match err {
                        CommitError::CommitConflict if attempt < max_retries => {
                            attempt += 1;
                            tokio::time::sleep(Duration::from_millis(10 * u64::from(attempt)))
                                .await;
                        }
                        CommitError::CommitConflict => {
                            return Err(NamespaceError::ConcurrentModification {
                                message: format!(
                                    "Dropping namespace '{}' conflicted after {max_retries} retries",
                                    self.root
                                ),
                            }
                            .into());
                        }
                        CommitError::OtherError(err) => return Err(err),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lance::dataset::builder::DatasetBuilder;
    use lance_namespace::LanceNamespace;
    use lance_namespace::models::{
        CreateNamespaceRequest, DeclareTableRequest, DescribeNamespaceRequest, DropTableRequest,
        ListTablesRequest, NamespaceExistsRequest,
    };
    use lance_testing::tempdir::TempStdDir;
    use tokio::sync::Barrier;

    use super::DELETED_METADATA_KEY;
    use crate::{DirectoryNamespace, DirectoryNamespaceBuilder};

    async fn namespace(root: &str) -> DirectoryNamespace {
        DirectoryNamespaceBuilder::new(root)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap()
    }

    fn declaration(id: &[&str]) -> DeclareTableRequest {
        DeclareTableRequest {
            id: Some(id.iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn lifecycle_tombstone_blocks_cached_and_new_connections() {
        let dir = TempStdDir::default();
        let root = dir.to_str().unwrap();
        let first = namespace(root).await;
        assert!(first.manifest_is_deleted().await.is_err());
        first.initialize_manifest().await.unwrap();
        let stale = namespace(root).await;
        stale.initialize_manifest().await.unwrap();
        assert!(!first.manifest_is_deleted().await.unwrap());
        assert!(first.drop_root_namespace().await.unwrap());
        assert!(!first.drop_root_namespace().await.unwrap());
        let reopened = namespace(root).await;
        for ns in [&first, &stale, &reopened] {
            assert!(ns.manifest_is_deleted().await.unwrap());
            assert!(ns.initialize_manifest().await.is_err());
            assert!(ns.declare_table(declaration(&["t"])).await.is_err());
            assert!(
                ns.list_tables(ListTablesRequest {
                    id: Some(vec![]),
                    ..Default::default()
                })
                .await
                .is_err()
            );
            assert!(
                ns.describe_namespace(DescribeNamespaceRequest {
                    id: Some(vec![]),
                    ..Default::default()
                })
                .await
                .is_err()
            );
            assert!(
                ns.namespace_exists(NamespaceExistsRequest {
                    id: Some(vec![]),
                    ..Default::default()
                })
                .await
                .is_err()
            );
        }
        let ds = DatasetBuilder::from_uri(format!("{root}/__manifest"))
            .load()
            .await
            .unwrap();
        assert_eq!(ds.metadata()[DELETED_METADATA_KEY], "true");
        assert_eq!(ds.version().version, 2);
    }

    #[tokio::test]
    async fn lifecycle_restrict_counts_unwritten_and_nested_tables() {
        let dir = TempStdDir::default();
        let ns = namespace(dir.to_str().unwrap()).await;
        ns.initialize_manifest().await.unwrap();
        ns.create_namespace(CreateNamespaceRequest {
            id: Some(vec!["child".into()]),
            ..Default::default()
        })
        .await
        .unwrap();
        ns.declare_table(declaration(&["child", "pending"]))
            .await
            .unwrap();
        assert!(
            ns.drop_root_namespace()
                .await
                .unwrap_err()
                .to_string()
                .contains("table registrations")
        );
        assert!(!ns.manifest_is_deleted().await.unwrap());
        ns.drop_table(DropTableRequest {
            id: Some(vec!["child".into(), "pending".into()]),
            ..Default::default()
        })
        .await
        .unwrap();
        assert!(ns.drop_root_namespace().await.unwrap());
    }

    #[tokio::test]
    async fn lifecycle_drop_races_table_declaration() {
        for _ in 0..12 {
            let dir = TempStdDir::default();
            let root = dir.to_str().unwrap();
            let dropper = namespace(root).await;
            dropper.initialize_manifest().await.unwrap();
            let creator = namespace(root).await;
            let barrier = Arc::new(Barrier::new(2));
            let (dropped, created) = tokio::join!(
                async {
                    barrier.wait().await;
                    dropper.drop_root_namespace().await
                },
                async {
                    barrier.wait().await;
                    creator.declare_table(declaration(&["t"])).await
                }
            );
            assert_ne!(dropped.is_ok(), created.is_ok());
            assert_eq!(
                dropper.manifest_is_deleted().await.unwrap(),
                dropped.is_ok()
            );
        }
    }
}
