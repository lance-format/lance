// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata guards for tagged fragment reuse histories.

use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;

use super::FRAG_REUSE_INDEX_NAME;
use crate::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
use crate::format::{IndexMetadata, Manifest};
use crate::io::commit::ManifestLocation;
use crate::io::manifest::read_manifest_indexes;

/// Whether an index entry requires the tagged FRI format contract.
pub fn is_tagged(index: &IndexMetadata) -> bool {
    index.name == FRAG_REUSE_INDEX_NAME && index.index_version != 0
}

/// Refuse publication of tagged history without both manifest capability bits.
pub fn validate_flags(manifest: &Manifest, indices: &[IndexMetadata]) -> Result<()> {
    if indices.iter().any(is_tagged)
        && (manifest.reader_feature_flags & manifest.writer_feature_flags)
            & FLAG_FRAGMENT_REUSE_INDEX
            == 0
    {
        return Err(Error::corrupt_file_named(
            "manifest",
            "tagged FRI metadata requires both reader and writer feature flags",
        ));
    }
    Ok(())
}

/// Cloning requires relocation of external mappings, which is not implemented yet.
/// A sticky flag alone need not mean that a mapping still exists.
pub async fn ensure_clone_supported(
    store: &ObjectStore,
    location: &ManifestLocation,
    manifest: &Manifest,
) -> Result<()> {
    if manifest.reader_feature_flags & FLAG_FRAGMENT_REUSE_INDEX == 0 {
        return Ok(());
    }
    let indices = read_manifest_indexes(store, location, manifest).await?;
    if indices.iter().any(is_tagged) {
        return Err(Error::not_supported(
            "Cloning tagged FRI requires row-map reference relocation. Please upgrade to a version supporting FRI clone",
        ));
    }
    Ok(())
}
