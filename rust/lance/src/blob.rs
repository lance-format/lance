// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Blob column builders, file-level writers, and planned reads through blob handles.
//!
//! Logical blob input uses either `Struct<data: LargeBinary?, uri: Utf8?>` or the complete
//! `Struct<data: LargeBinary?, uri: Utf8?, position: UInt64?, size: UInt64?>` shape. In the
//! complete shape, `position` and `size` select a non-empty range within an external `uri` and
//! must be set together. Every non-null row must set exactly one of `data` and `uri`. File-level
//! blob preparation produces a kind-aware writer intermediate with `blob_id` and range fields.
//!
//! Managed blob writes and compaction reuse are opt-in. Set Arrow schema metadata
//! `lance.blob.managed` to `true` to enable them. Committing this opt-in
//! permanently requires Managed-aware readers and maintenance clients. Without
//! this metadata, logical writes use Inline, Packed, and Dedicated descriptors;
//! prepared Managed descriptors are rejected.

pub mod reference;

pub use lance_table_runtime::blob::{
    BlobArrayBuilder, BlobDescriptor, BlobDescriptorArrayBuilder, BlobDescriptorColumn,
    BlobFieldOptions, BlobFile, BlobRange, BlobReadRange, BlobWriteSources, DedicatedBlobWriter,
    ManagedBlobWriter, PackedBlobWriter, ReadBlobFile, ReadBlobFilesBuilder, ReadBlobFilesStream,
    blob_field, blob_field_with_options, read_blob_file_ranges, read_blob_files,
};

pub(crate) use lance_table_runtime::blob::{
    BlobIdAllocator, blob_v2_layout, logical_to_prepared_blob_field,
    logical_to_prepared_blob_schema, prepared_to_descriptor_blob_schema,
};
