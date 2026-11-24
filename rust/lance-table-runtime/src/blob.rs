// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lazy handles and planned reads for blob payloads stored by a table.

mod clone;
mod external;
mod file;
#[doc(hidden)]
pub mod locate;
mod managed;
mod materialize;
mod prepared;
mod read;
mod rewrite;
mod sidecar;
mod types;
mod write;
mod write_sources;

#[doc(hidden)]
pub use clone::copy_blob_columns;
pub(crate) use external::ExternalBaseCandidate;
#[doc(hidden)]
pub use external::{ExternalBaseResolver, validate_external_blob_references};
pub use managed::preserve_managed_descriptors;
#[doc(hidden)]
pub use managed::{field_contains_blob, managed_references};
pub use materialize::{
    BlobMaterializationAdmission, BlobMaterializationContext, MaterializedBlobBatch,
    blob_v2_descriptor_schema, materialize_blob_v2_binary_batch,
    materialize_blob_v2_binary_batch_with_admission, materialize_blob_v2_binary_batch_with_context,
    public_blob_v2_binary_output_schema, schema_has_blob_v2_binary_view, split_batch_by_bytes,
};
pub use prepared::{prepared_blob_batch_to_descriptors, prepared_blob_ids, prepared_sidecar_bytes};
pub use rewrite::{BlobV2BatchRewritePlan, field_contains_blob_v2, transform_blob_v2_batch};
#[doc(hidden)]
pub use sidecar::SidecarRewrite;
#[doc(hidden)]
pub use types::BlobIdAllocator;
pub use types::{
    BlobArrayBuilder, BlobDescriptor, BlobDescriptorArrayBuilder, BlobDescriptorColumn,
    BlobFieldOptions, BlobRange, DedicatedBlobWriter, ManagedBlobWriter, PackedBlobWriter,
    blob_field, blob_field_with_options,
};
#[doc(hidden)]
pub use types::{
    blob_v2_layout, blob_v2_shape_error, logical_to_prepared_blob_field,
    logical_to_prepared_blob_schema, prepared_blob_field_with_metadata,
    prepared_to_descriptor_blob_schema, prepared_to_logical_blob_schema,
    sidecar_path_for_data_file, validate_prepared_blob_array,
};
pub use write::{BlobPayloadInventory, BlobPreprocessor};
pub use write_sources::BlobWriteSources;
#[cfg(test)]
mod test_util;

pub use file::BlobFile;
use locate::collect_blob_v2_descriptor_files;
pub use read::{
    BlobRangeRequest, BlobReadRange, ReadBlob, ReadBlobFile, ReadBlobFilesBuilder,
    ReadBlobFilesStream, ReadBlobRange, ReadBlobRangesBuilder, ReadBlobRangesStream,
    ReadBlobsBuilder, ReadBlobsStream, read_blob_file_ranges, read_blob_files,
};

/// Payload execution shared with callers that already hold descriptors.
#[doc(hidden)]
pub mod execution {
    pub use super::read::{
        BlobEntry, IndexedReadBlob, ReadBlobsExecution, execute_blob_entries_with_execution,
    };
}

#[doc(hidden)]
pub mod thresholds;
