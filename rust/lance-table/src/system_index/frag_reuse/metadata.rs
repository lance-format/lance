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

/// Whether the table's fragment reuse history is governed by the tagged
/// format contract, which decides the record form every FRI write must
/// take: append transitions, never a v0 snapshot replacement.
///
/// The sticky [`FLAG_FRAGMENT_REUSE_INDEX`] is the final authority: the
/// first tagged commit sets it and nothing ever clears it -- a trim may
/// delete a fully drained entry, but the flag survives -- so a
/// flag-carrying manifest stays tagged even when no entry currently
/// exists. The entry arm keeps the decision correct for a tagged entry
/// observed before the flag is stamped on (a manifest assembled
/// mid-commit); a v0 entry under the flag does NOT make the table v0: the
/// next tagged commit lifts it byte-verbatim instead.
pub fn uses_tagged_fri(manifest: &Manifest, existing_fri_entry: Option<&IndexMetadata>) -> bool {
    manifest.reader_feature_flags & FLAG_FRAGMENT_REUSE_INDEX != 0
        || existing_fri_entry.is_some_and(is_tagged)
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

    #[test]
    fn sticky_flag_or_tagged_entry_makes_the_table_tagged() {
        let mut flagged = crate::transaction::test_support::sample_manifest();
        flagged.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        let unflagged = crate::transaction::test_support::sample_manifest();
        // The flag alone decides, entry present or not, v0 entry included.
        assert!(uses_tagged_fri(&flagged, None));
        assert!(uses_tagged_fri(
            &flagged,
            Some(&entry(FRAG_REUSE_INDEX_NAME, 0))
        ));
        // Without the flag, only a tagged entry does.
        assert!(uses_tagged_fri(
            &unflagged,
            Some(&entry(FRAG_REUSE_INDEX_NAME, 1))
        ));
        assert!(!uses_tagged_fri(
            &unflagged,
            Some(&entry(FRAG_REUSE_INDEX_NAME, 0))
        ));
        assert!(!uses_tagged_fri(&unflagged, None));
    }
}
