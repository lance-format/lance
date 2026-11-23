// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use object_store::path::Path;

use super::BlobFile;
use crate::blob::BlobDescriptor;
use crate::dataset::BLOBS_DIR;
use crate::{Error, Result};
use lance_core::utils::blob::validate_managed_reference;

/// Trusted payload sources for a single prepared Blob v2 batch.
///
/// Add each source and put the returned descriptor in the corresponding leaf
/// using [`crate::BlobDescriptorArrayBuilder`]. These descriptors are local to
/// the batch and must be written with
/// [`crate::dataset::fragment::FileFragment::write_columns_with_blob_sources`]
/// or [`crate::Dataset::write_data_file_part_with_blob_sources`]. The writer
/// replaces them with ordinary persisted Blob descriptors before encoding.
/// Only trusted code should build these arrays and resolve the payloads.
#[derive(Debug, Default)]
pub struct BlobWriteSources {
    sources: Vec<BlobWriteObject>,
}

#[derive(Debug)]
pub(super) enum BlobWriteObject {
    Upload { relative_path: Path, size: u64 },
    Stored(BlobFile),
}

impl BlobWriteSources {
    /// Register a completed immutable payload under the destination base.
    ///
    /// `relative_path` has the form `_blobs/<namespace>/<attempt>/<object>.payload`.
    /// It resolves through the destination's object store and base, including
    /// when the table uses ordinary Blob v2. The caller must confirm upload
    /// completion and integrity before registering the object and keep it
    /// available through the write. The writer verifies the exact object size.
    /// On a Managed-enabled table the object becomes the backing payload of the
    /// new descriptor. On an ordinary table Lance writes a destination-owned
    /// copy instead.
    ///
    /// ```
    /// use lance::blob::BlobWriteSources;
    /// use object_store::path::Path;
    ///
    /// # fn example() -> lance::Result<()> {
    /// let mut sources = BlobWriteSources::default();
    /// let descriptor = sources.add_uploaded_blob(
    ///     Path::from("_blobs/functions/attempt-12/value.payload"), 1024,
    /// )?;
    /// # let _ = descriptor;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_uploaded_blob(&mut self, relative_path: Path, size: u64) -> Result<BlobDescriptor> {
        let relative_path = validate_managed_reference(relative_path.as_ref(), 0, size)?;
        let mut parts = relative_path.parts();
        if !parts.next().is_some_and(|part| part.as_ref() == BLOBS_DIR)
            || parts.count() != 3
            || relative_path.extension() != Some("payload")
        {
            return Err(Error::invalid_input(format!(
                "Blob upload path '{relative_path}' must match _blobs/<namespace>/<attempt>/<object>.payload"
            )));
        }
        self.add(
            BlobWriteObject::Upload {
                relative_path,
                size,
            },
            size,
        )
    }

    /// Register the resolved range of an existing Blob, for example when
    /// carrying an ordinary sidecar forward from a staged file. Lance copies
    /// the range using the source's store. The caller must retain it until the
    /// copy completes.
    pub fn add_blob_file(&mut self, file: BlobFile) -> Result<BlobDescriptor> {
        let size = file.size();
        self.add(BlobWriteObject::Stored(file), size)
    }

    fn add(&mut self, source: BlobWriteObject, size: u64) -> Result<BlobDescriptor> {
        let id = self
            .sources
            .len()
            .checked_add(1)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or_else(|| Error::invalid_input("Too many Blob write sources in one batch"))?;
        self.sources.push(source);
        Ok(BlobDescriptor::Packed {
            blob_id: id,
            offset: 0,
            size,
        })
    }

    pub(in crate::dataset) fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub(super) fn get(&self, id: u32, offset: u64, size: u64) -> Result<&BlobWriteObject> {
        let source = id
            .checked_sub(1)
            .and_then(|index| self.sources.get(index as usize))
            .ok_or_else(|| Error::invalid_input(format!("Unknown Blob write source ID {id}")))?;
        let expected_size = match source {
            BlobWriteObject::Upload { size, .. } => *size,
            BlobWriteObject::Stored(file) => file.size(),
        };
        if offset != 0 || size != expected_size {
            return Err(Error::invalid_input(format!(
                "Blob write source ID {id} must have offset 0 and size {expected_size}, got offset {offset} and size {size}"
            )));
        }
        Ok(source)
    }
}
