// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use lance::dataset::Dataset;
use lance::dataset::transaction::{Operation, Transaction};
use lance_core::Result;
use lance_namespace::error::NamespaceError;
use lance_namespace::models::DescribeTransactionResponse;
use object_store::path::Path;
use object_store::{Error as ObjectStoreError, ObjectStoreExt};

use super::DirectoryNamespace;

/// Persistent record of `alter_transaction` outcomes for a single transaction.
///
/// Lance's transaction file is immutable once written, so we record any
/// modifications (status transitions, extra properties, tombstoned properties)
/// in a namespace-owned sidecar file. The sidecar is then merged into the
/// response of subsequent `describe_transaction` / `alter_transaction` calls.
///
/// Serialization is implemented manually via `serde_json::Value` to avoid
/// pulling in `serde`'s `derive` feature for this crate.
#[derive(Debug, Clone, Default)]
pub(super) struct TransactionAlteration {
    /// The most recently applied status, if any.
    pub(super) status: Option<String>,
    /// User-defined properties layered on top of the immutable transaction
    /// properties. Values here take precedence over the transaction's own
    /// properties when both are present.
    pub(super) properties: HashMap<String, String>,
    /// Names of transaction properties that have been tombstoned via
    /// `unset_property_action`. A tombstoned key is hidden from the response
    /// even when the immutable transaction still carries it.
    pub(super) removed_properties: std::collections::HashSet<String>,
}

impl TransactionAlteration {
    /// JSON field names used for the sidecar on-disk representation.
    const F_STATUS: &'static str = "status";
    const F_PROPERTIES: &'static str = "properties";
    const F_REMOVED_PROPERTIES: &'static str = "removed_properties";

    /// Serialize this alteration to a JSON byte vector.
    ///
    /// Uses the same pattern as `dir/manifest.rs`: rely on the built-in
    /// `Serialize` impls for `Option<String>`, `HashMap<String, String>` and
    /// `HashSet<String>` provided by the `serde` crate (transitively pulled in
    /// by `serde_json`), so no `serde` derive nor extra dependency is needed.
    fn to_json_bytes(&self) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec(&serde_json::json!({
            Self::F_STATUS: self.status,
            Self::F_PROPERTIES: self.properties,
            Self::F_REMOVED_PROPERTIES: self.removed_properties,
        }))
    }

    /// Deserialize an alteration from JSON bytes, mirroring the
    /// `serde_json::from_slice::<HashMap<String, String>>(...)` idiom already
    /// used in `dir/manifest.rs`. Missing / null fields fall back to defaults
    /// so that the sidecar format stays forward-compatible.
    fn from_json_slice(bytes: &[u8]) -> serde_json::Result<Self> {
        let mut obj: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(bytes)?;
        Ok(Self {
            status: serde_json::from_value(
                obj.remove(Self::F_STATUS)
                    .unwrap_or(serde_json::Value::Null),
            )?,
            properties: serde_json::from_value(
                obj.remove(Self::F_PROPERTIES)
                    .unwrap_or(serde_json::Value::Null),
            )
            .unwrap_or_default(),
            removed_properties: serde_json::from_value(
                obj.remove(Self::F_REMOVED_PROPERTIES)
                    .unwrap_or(serde_json::Value::Null),
            )
            .unwrap_or_default(),
        })
    }
}

impl DirectoryNamespace {
    fn transaction_operation_name(transaction: &Transaction) -> String {
        match &transaction.operation {
            Operation::CreateIndex {
                new_indices,
                removed_indices,
                ..
            } if new_indices.is_empty() && !removed_indices.is_empty() => "DropIndex".to_string(),
            _ => transaction.operation.to_string(),
        }
    }

    pub(super) fn transaction_response(
        version: u64,
        transaction: &Transaction,
        alteration: Option<TransactionAlteration>,
    ) -> DescribeTransactionResponse {
        let mut properties = transaction
            .transaction_properties
            .as_ref()
            .map(|properties| (**properties).clone())
            .unwrap_or_default();

        // Apply persisted alterations on top of the immutable transaction
        // properties so callers see the current effective state.
        let mut effective_status = "SUCCEEDED".to_string();
        if let Some(alteration) = alteration {
            for key in &alteration.removed_properties {
                properties.remove(key);
            }
            for (key, value) in alteration.properties {
                properties.insert(key, value);
            }
            if let Some(status) = alteration.status {
                effective_status = status;
            }
        }

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
            status: effective_status,
            properties: Some(properties),
            ..Default::default()
        }
    }

    /// When transaction_id is not parseable as a version number (i.e. it's a UUID),
    /// find_transaction iterates through every version in reverse, reading each
    /// transaction file from storage. For tables with many versions this will
    /// be extremely slow — each iteration is a separate I/O call.
    pub(super) async fn find_transaction(
        &self,
        dataset: &Dataset,
        id: &str,
    ) -> Result<(u64, Transaction)> {
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

    /// Relative directory (under a table's Lance root) used to persist
    /// alter_transaction outcomes. The Lance transaction file itself is
    /// immutable, so we keep alterations in a namespace-owned sidecar.
    const TRANSACTION_ALTERATIONS_DIR: &'static str = "_alter_transactions";

    fn transaction_alteration_path(&self, table_uri: &str, txn_uuid: &str) -> Result<Path> {
        let table_path = self.object_store_path_from_uri(table_uri)?;
        Ok(table_path
            .join(Self::TRANSACTION_ALTERATIONS_DIR)
            .join(format!("{}.json", txn_uuid).as_str()))
    }

    pub(super) async fn load_transaction_alteration(
        &self,
        table_uri: &str,
        txn_uuid: &str,
    ) -> Result<Option<TransactionAlteration>> {
        let path = self.transaction_alteration_path(table_uri, txn_uuid)?;
        match self.object_store.inner.get(&path).await {
            Ok(get_result) => {
                let bytes = get_result.bytes().await.map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!(
                            "Failed to read alter_transaction sidecar for '{}': {}",
                            txn_uuid, e
                        ),
                    })
                })?;
                let alteration = TransactionAlteration::from_json_slice(&bytes).map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!(
                            "Failed to parse alter_transaction sidecar for '{}': {}",
                            txn_uuid, e
                        ),
                    })
                })?;
                Ok(Some(alteration))
            }
            Err(ObjectStoreError::NotFound { .. }) => Ok(None),
            Err(e) => Err(lance_core::Error::from(NamespaceError::Internal {
                message: format!(
                    "Failed to load alter_transaction sidecar for '{}': {}",
                    txn_uuid, e
                ),
            })),
        }
    }

    pub(super) async fn save_transaction_alteration(
        &self,
        table_uri: &str,
        txn_uuid: &str,
        alteration: &TransactionAlteration,
    ) -> Result<()> {
        let path = self.transaction_alteration_path(table_uri, txn_uuid)?;
        let bytes = alteration.to_json_bytes().map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!(
                    "Failed to serialize alter_transaction sidecar for '{}': {}",
                    txn_uuid, e
                ),
            })
        })?;
        self.object_store
            .inner
            .put(&path, bytes.into())
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to persist alter_transaction sidecar for '{}': {}",
                        txn_uuid, e
                    ),
                })
            })?;
        Ok(())
    }
}
