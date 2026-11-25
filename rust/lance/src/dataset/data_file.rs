// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::{Result, datatypes::Schema};
use lance_file::version::ConcreteFileVersion;
use lance_table_runtime::write::DataFileTarget as RuntimeDataFileTarget;
#[cfg(test)]
use object_store::path::Path;
use serde::{Deserialize, Serialize};

use super::Dataset;

/// Serializable identity and logical schema of a final concatenated data file.
///
/// Reuse the same identity for every part write and final concatenation. Callers
/// serialize and deserialize this value with serde. The representation preserves
/// field IDs, metadata, and loaded dictionary values. The caller owns checkpoint
/// state and must keep every use associated with the same dataset and resolved base;
/// Lance does not validate that association across [`Dataset`] instances.
/// Checkpoints must come from trusted application state. Deserialization does not
/// prove artifact ownership or establish whether a target has been committed.
/// Callers must fence stale workers and never resume writes or assembly for a
/// committed target; only staging cleanup via [`Self::finish`] remains valid.
///
/// ```
/// # use lance::dataset::DataFileTarget;
/// # fn checkpoint(target: &DataFileTarget) -> Result<(), serde_json::Error> {
/// let bytes = serde_json::to_vec(target)?;
/// let restored: DataFileTarget = serde_json::from_slice(&bytes)?;
/// # let _ = restored;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DataFileTarget(pub(super) RuntimeDataFileTarget);

impl std::fmt::Debug for DataFileTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl DataFileTarget {
    /// Create a final data-file target with Lance's ordinary random file naming.
    ///
    /// This only creates a runtime identity; it does not create, reserve, or
    /// register an object. The caller owns the target lifetime, part storage,
    /// cleanup, and commit state. Prepared Blob v2 schemas are normalized to
    /// their caller-visible logical form; the persisted descriptor schema remains
    /// an internal writer detail.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance::dataset::DataFileTarget;
    /// use lance_core::datatypes::Schema;
    /// use lance_file::version::ConcreteFileVersion;
    ///
    /// # fn target(schema: Arc<Schema>) -> lance_core::Result<DataFileTarget> {
    /// DataFileTarget::new(
    ///     None,
    ///     schema,
    ///     ConcreteFileVersion::V2_2,
    /// )
    /// # }
    /// ```
    pub fn new(
        base_id: Option<u32>,
        schema: Arc<Schema>,
        version: ConcreteFileVersion,
    ) -> Result<Self> {
        RuntimeDataFileTarget::new(base_id, schema, version).map(Self)
    }

    #[cfg(test)]
    pub(super) fn data_file_key(&self) -> &str {
        self.0.data_file_key()
    }

    #[cfg(test)]
    pub(super) fn object_path(&self, data_dir: &Path) -> Path {
        self.0.object_path(data_dir)
    }

    /// Release staging parts after a successful commit, keeping the final file
    /// and its managed Blob payloads. Stop all writers and assemblers first.
    ///
    /// This also removes incomplete parts. Missing files are OK; storage failures
    /// may leave partial cleanup and can be retried. Use the same dataset and
    /// resolved base as the original write. This does not commit the target.
    /// All Blob payloads in the target's namespace are retained, including those
    /// from failed or unused retry leases. Ordinary dataset GC also retains these
    /// payloads while the parent data file is referenced; this operation does not
    /// perform per-Blob reachability collection.
    ///
    /// ```
    /// # use lance::{Dataset, dataset::DataFileTarget};
    /// # async fn release(dataset: &Dataset, target: &DataFileTarget) -> lance_core::Result<()> {
    /// // After committing the assembled file and releasing all part users:
    /// target.finish(dataset).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn finish(&self, dataset: &Dataset) -> Result<()> {
        Box::pin(self.0.finish(&dataset.table_view().to_table())).await
    }

    /// Delete an abandoned target's staging parts, final file, and file-relative
    /// Packed/Dedicated sidecars, including those left by failed writes or assembly.
    /// Independent Managed objects are not owned by this target; unreferenced
    /// objects follow the dataset's ordinary garbage-collection policy.
    ///
    /// The caller must stop all users and ensure no current or retained dataset
    /// version, checkpoint, or future commit needs this target. Lance does not
    /// track task liveness or check historical references. Use the original
    /// dataset/base mapping. Missing objects are OK; retry on storage failures.
    ///
    /// ```
    /// # use lance::{Dataset, dataset::DataFileTarget};
    /// # async fn abandon(dataset: &Dataset, target: &DataFileTarget) -> lance_core::Result<()> {
    /// // After abandoning the target and stopping all its users:
    /// target.cleanup(dataset).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn cleanup(&self, dataset: &Dataset) -> Result<()> {
        Box::pin(self.0.cleanup(&dataset.table_view().to_table())).await
    }
}
