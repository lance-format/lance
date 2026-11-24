// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lazy handles and planned reads for blob payloads stored by a table.

mod file;
#[doc(hidden)]
pub mod locate;
mod read;
#[cfg(test)]
mod test_util;

pub use file::BlobFile;
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
