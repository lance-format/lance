// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compatibility adapter for the original blob-specific retention API.

use lance_core::{Error, Result};
use prost::Message;

use super::{BlobSourceContext, pb};
use crate::Dataset;
use crate::dataset::retention::SnapshotRetention;

/// Blob-context convenience adapter for general snapshot retention.
///
/// Ordinary queries should use [`SnapshotRetention`] with a
/// [`SnapshotIdentity`](crate::dataset::snapshot::SnapshotIdentity). This adapter
/// preserves the original context transport and delegates protection to the same
/// snapshot GC root. Acquire before issuing references and release after every
/// consumer and retry has stopped. Dropping a token never releases its tag.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::blob::reference::{BlobSourceContext, BlobSnapshotRetention};
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let context = BlobSourceContext::from_dataset(dataset, "catalog/table", "workers").await?;
/// let retention = BlobSnapshotRetention::acquire(dataset, context, "unique-query-id").await?;
/// retention.release(dataset).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct BlobSnapshotRetention {
    context: BlobSourceContext,
    retention: SnapshotRetention,
}

impl BlobSnapshotRetention {
    /// Acquire the source snapshot using a globally unique query-lifetime ID.
    pub async fn acquire(
        dataset: &Dataset,
        context: BlobSourceContext,
        retention_id: impl AsRef<str>,
    ) -> Result<Self> {
        context.validate_dataset(dataset)?;
        let retention =
            SnapshotRetention::acquire(dataset, context.snapshot_identity()?, retention_id).await?;
        Ok(Self { context, retention })
    }
    /// The original blob context, including worker routing and capabilities.
    pub fn context(&self) -> &BlobSourceContext {
        &self.context
    }
    /// Protected snapshot tag owned by this query.
    pub fn tag_name(&self) -> &str {
        self.retention.tag_name()
    }
    /// Encode the original blob-context token without storage credentials.
    pub fn encode(&self) -> Vec<u8> {
        pb::SnapshotRetention {
            protocol_version: 1,
            context: self.context.encode().into(),
            tag_name: self.tag_name().to_string(),
        }
        .encode_to_vec()
    }
    /// Decode a token without acquiring or renewing protection.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let token = pb::SnapshotRetention::decode(bytes).map_err(|error| {
            Error::invalid_input(format!("Invalid blob retention token: {error}"))
        })?;
        if token.protocol_version != 1 {
            return Err(Error::invalid_input(format!(
                "Unsupported retention protocol version {}",
                token.protocol_version
            )));
        }
        let context = BlobSourceContext::decode(&token.context)?;
        let retention =
            SnapshotRetention::from_parts(context.snapshot_identity()?, token.tag_name)?;
        Ok(Self { context, retention })
    }
    /// Idempotently release after all workers, reads, and retries have stopped.
    pub async fn release(&self, dataset: &Dataset) -> Result<()> {
        self.retention.release(dataset).await
    }
}
