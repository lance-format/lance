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

/// Deep-cloning a tagged history would need every referenced row-map file
/// copied and its stable-partition references relocated, which no writer
/// implements. Shallow clone does not consult this: it relocates the
/// references in the `lance` crate instead (the decode machinery lives
/// there), rejecting on its own any content it cannot fully interpret.
/// A sticky flag alone need not mean that a mapping still exists.
pub async fn ensure_deep_clone_supported(
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
            "Deep-cloning tagged FRI requires copying row maps and relocating their references. Please upgrade to a version supporting FRI deep clone",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn entry(name: &str, index_version: i32) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::nil(),
            fields: vec![],
            covering_fields: vec![],
            name: name.into(),
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: None,
            index_version,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    #[test]
    fn tagged_requires_the_reserved_name_and_a_nonzero_version() {
        assert!(!is_tagged(&entry(FRAG_REUSE_INDEX_NAME, 0)));
        assert!(is_tagged(&entry(FRAG_REUSE_INDEX_NAME, 1)));
        assert!(is_tagged(&entry(FRAG_REUSE_INDEX_NAME, 2)));
        assert!(!is_tagged(&entry("user_idx", 0)));
        assert!(!is_tagged(&entry("user_idx", 1)));
    }
}
