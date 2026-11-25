// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod clone;
#[cfg(test)]
mod compaction;
mod managed;
mod materialize;
mod rewrite;
#[cfg(test)]
mod rewrite_tests;
#[cfg(test)]
pub mod test_util;
#[cfg(test)]
mod write;

#[cfg(test)]
mod file_tests;
#[cfg(test)]
mod locate_tests;
#[cfg(test)]
mod read_tests;

use lance_table_runtime::blob::locate;

pub use lance_table_runtime::blob::{
    BlobFile, BlobPayloadInventory, BlobPreprocessor, BlobRangeRequest, BlobReadRange, ReadBlob,
    ReadBlobRange, ReadBlobRangesBuilder, ReadBlobRangesStream, ReadBlobsBuilder, ReadBlobsStream,
    prepared_blob_batch_to_descriptors, prepared_blob_ids, prepared_sidecar_bytes,
};
pub(super) use lance_table_runtime::blob::{
    ExternalBaseResolver, SidecarRewrite, field_contains_blob, validate_external_blob_references,
};
pub(super) use locate::data_file_key_from_path;
pub use managed::preserve_managed_descriptors;
pub(super) use managed::{managed_paths, may_be_same_store};
pub use materialize::{
    BlobMaterializationAdmission, BlobMaterializationContext, MaterializedBlobBatch,
    blob_v2_descriptor_schema, materialize_blob_v2_binary_batch,
    materialize_blob_v2_binary_batch_with_admission, public_blob_v2_binary_output_schema,
    schema_has_blob_v2_binary_view, split_batch_by_bytes,
};
pub use rewrite::{BlobV2BatchRewritePlan, field_contains_blob_v2, transform_blob_v2_batch};
