// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use lance_core::{Result, datatypes::Schema};
use lance_table_runtime::blob::BlobV2BatchRewritePlan as RuntimeRewritePlan;

use crate::Dataset;

pub use lance_table_runtime::blob::field_contains_blob_v2;

/// Used by compaction to preserve payload references or materialize payload bytes.
pub struct BlobV2BatchRewritePlan(RuntimeRewritePlan);

impl BlobV2BatchRewritePlan {
    /// Build a plan from the dataset schema and scanned batch schema.
    ///
    /// Input must include `_rowaddr`; Struct children must be complete and in
    /// dataset order. Scan legacy blobs as binary and Blob v2 as descriptors.
    /// `keep_row_addr` retains `_rowaddr` in the output.
    ///
    /// `preserve_managed` keeps payload references instead of materializing bytes.
    /// It requires Managed blobs and a V2_2+ writer. Write output to the same
    /// dataset's primary storage.
    pub fn try_new(
        schema: &Schema,
        input_schema: &ArrowSchema,
        keep_row_addr: bool,
        preserve_managed: bool,
    ) -> Result<Self> {
        RuntimeRewritePlan::try_new(schema, input_schema, keep_row_addr, preserve_managed).map(Self)
    }

    #[cfg(test)]
    pub(in crate::dataset) fn with_sidecar_rewrite(
        self,
        sidecars: Arc<super::SidecarRewrite>,
    ) -> Self {
        Self(self.0.with_sidecar_rewrite(sidecars))
    }

    /// Schema of rewritten batches, for opening the destination writer.
    pub fn output_schema(&self) -> &SchemaRef {
        self.0.output_schema()
    }

    /// Rewrite a batch using Blob payloads from `dataset`.
    ///
    /// `batch` must match the planned schema and contain valid, non-null `UInt64`
    /// row addresses from this dataset snapshot. These preconditions are not
    /// validated. Keep the snapshot alive until the output is committed.
    ///
    /// May read from storage. Batches can be transformed concurrently.
    pub async fn transform_batch(
        &self,
        dataset: &Arc<Dataset>,
        batch: RecordBatch,
    ) -> Result<RecordBatch> {
        let table = dataset.table_view().to_table();
        Box::pin(self.0.transform_batch(&table, batch)).await
    }
}

/// Prepare Blob v2 values in a batch for rewriting.
pub async fn transform_blob_v2_batch(
    dataset: &Arc<Dataset>,
    schema: &Schema,
    batch: RecordBatch,
    keep_row_addr: bool,
) -> Result<RecordBatch> {
    let table = dataset.table_view().to_table();
    Box::pin(lance_table_runtime::blob::transform_blob_v2_batch(
        &table,
        schema,
        batch,
        keep_row_addr,
    ))
    .await
}
