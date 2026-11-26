// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::{FutureExt, future::BoxFuture};
use lance_core::{Result, datatypes::Schema};
use lance_table_runtime::blob as runtime;

use crate::Dataset;

pub use runtime::{
    BlobMaterializationAdmission, BlobMaterializationContext, BlobMaterializationReservation,
    MaterializedBlobBatch, blob_v2_descriptor_schema, public_blob_v2_binary_output_schema,
    schema_has_blob_v2_binary_view, split_batch_by_bytes,
};

/// Materialize blob v2 descriptor arrays in a decoded batch into binary arrays.
///
/// The input batch must include `_rowaddr`, which is used to resolve packed,
/// dedicated, inline, and external blob payload locations. `output_schema`
/// defines the exact returned columns, including requested system columns, with
/// blob v2 binary leaves exposed as plain `LargeBinary` fields.
pub async fn materialize_blob_v2_binary_batch(
    dataset: &Arc<Dataset>,
    output_schema: &Schema,
    batch: RecordBatch,
) -> Result<RecordBatch> {
    let table = dataset.table_view().to_table();
    Box::pin(runtime::materialize_blob_v2_binary_batch(
        &table,
        output_schema,
        batch,
    ))
    .await
}

pub fn materialize_blob_v2_binary_batch_with_admission<'a>(
    dataset: &'a Arc<Dataset>,
    output_schema: &'a Schema,
    batch: RecordBatch,
    context: &'a Arc<BlobMaterializationContext>,
    admission: BlobMaterializationAdmission,
) -> BoxFuture<'a, Result<MaterializedBlobBatch>> {
    async move {
        let table = dataset.table_view().to_table();
        runtime::materialize_blob_v2_binary_batch_with_admission(
            &table,
            output_schema,
            batch,
            context,
            admission,
        )
        .await
    }
    .boxed()
}
