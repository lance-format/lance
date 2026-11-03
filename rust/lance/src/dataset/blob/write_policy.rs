// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Application admission for newly supplied values, before blob preparation.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use lance_core::Result;

use crate::dataset::Dataset;
use datafusion::execution::SendableRecordBatchStream;

/// Server-owned context for resolving references in an incoming batch.
#[derive(Clone, Debug, Default)]
pub struct BlobWriteContext {
    /// Existing destination fields, including their blob extension metadata.
    pub target_schema: Option<Arc<ArrowSchema>>,
    /// Exact registered base IDs and their absolute locations.
    pub bases: HashMap<u32, String>,
}

impl BlobWriteContext {
    /// Build context from the destination snapshot rather than input metadata.
    pub fn for_dataset(dataset: Option<&Dataset>) -> Self {
        Self {
            target_schema: dataset.map(|dataset| Arc::new(dataset.schema().into())),
            bases: dataset
                .map(|dataset| {
                    dataset
                        .manifest()
                        .base_paths
                        .iter()
                        .map(|(id, base)| (*id, base.path.clone()))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

/// An application's policy for admitting newly supplied blob values.
///
/// The policy is runtime state: it is never read from Arrow metadata or persisted
/// in a manifest. Mutation implementations must apply it to incoming values,
/// before combining them with unchanged values from existing rows.
#[async_trait]
pub trait BlobWritePolicy: std::fmt::Debug + Send + Sync {
    /// Describe the normalized input, before schema compatibility is checked.
    fn schema(&self, input: &ArrowSchema, context: &BlobWriteContext) -> Result<ArrowSchema>;

    /// Normalize and authorize a batch before it reaches a storage writer.
    async fn apply(&self, batch: RecordBatch, context: &BlobWriteContext) -> Result<RecordBatch>;
}

/// Wrap an input stream without collecting it or changing commit semantics.
pub fn guard_blob_write_stream(
    stream: SendableRecordBatchStream,
    policy: Option<Arc<dyn BlobWritePolicy>>,
    context: BlobWriteContext,
) -> Result<SendableRecordBatchStream> {
    let Some(policy) = policy else {
        return Ok(stream);
    };
    let schema = Arc::new(policy.schema(stream.schema().as_ref(), &context)?);
    let stream = stream.then(move |batch| {
        let policy = policy.clone();
        let context = context.clone();
        async move {
            let batch = batch?;
            policy
                .apply(batch, &context)
                .await
                .map_err(|error| datafusion::error::DataFusionError::External(Box::new(error)))
        }
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
}
