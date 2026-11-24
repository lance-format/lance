// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod clone;
mod compaction;
mod external;
mod managed;
mod materialize;
mod prepared;
mod rewrite;
#[cfg(test)]
mod rewrite_tests;
#[cfg(test)]
pub mod test_util;
mod write;
mod write_sources;

#[cfg(test)]
mod file_tests;
#[cfg(test)]
mod locate_tests;
#[cfg(test)]
mod read_tests;

use lance_table_runtime::blob::locate;

pub(super) use compaction::SidecarRewrite;
pub(super) use external::{
    ExternalBaseCandidate, ExternalBaseResolver, validate_external_blob_references,
};
pub use lance_table_runtime::blob::{
    BlobFile, BlobRangeRequest, BlobReadRange, ReadBlob, ReadBlobFile, ReadBlobFilesBuilder,
    ReadBlobFilesStream, ReadBlobRange, ReadBlobRangesBuilder, ReadBlobRangesStream,
    ReadBlobsBuilder, ReadBlobsStream, read_blob_file_ranges, read_blob_files,
};
pub(super) use locate::{collect_blob_v2_descriptor_files, data_file_key_from_path};
pub use managed::preserve_managed_descriptors;
pub(super) use managed::{field_contains_blob, managed_paths, may_be_same_store};
pub use materialize::{
    BlobMaterializationAdmission, BlobMaterializationContext, MaterializedBlobBatch,
    blob_v2_descriptor_schema, materialize_blob_v2_binary_batch,
    materialize_blob_v2_binary_batch_with_admission, public_blob_v2_binary_output_schema,
    schema_has_blob_v2_binary_view, split_batch_by_bytes,
};
pub use prepared::{prepared_blob_batch_to_descriptors, prepared_blob_ids, prepared_sidecar_bytes};
pub use rewrite::{BlobV2BatchRewritePlan, field_contains_blob_v2, transform_blob_v2_batch};
pub use write::BlobPayloadInventory;
pub use write::BlobPreprocessor;
pub(super) use write::{
    blob_dedicated_threshold_from_metadata, blob_inline_threshold_from_metadata,
    blob_pack_file_threshold_from_metadata,
};
pub use write_sources::BlobWriteSources;
