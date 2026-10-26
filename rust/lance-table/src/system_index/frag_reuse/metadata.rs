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

/// Guard for a clone workflow that cannot relocate fragment reuse mapping
/// references: refuses to clone a manifest whose fragment reuse history is
/// tagged, because its mapping references would still point at the source
/// dataset after the copy.
///
/// The dataset clone APIs (`Dataset::shallow_clone` and `Dataset::deep_clone`
/// in the `lance` crate) relocate those references as part of the clone and
/// do not call this guard. It is kept, unchanged, for external callers that
/// still copy a dataset through a path that cannot relocate references; such
/// a path must keep rejecting tagged history, so this is not an unconditional
/// success.
///
/// A sticky [`FLAG_FRAGMENT_REUSE_INDEX`] alone need not mean that a mapping
/// still exists, so the index section is consulted before rejecting.
#[deprecated(
    since = "13.0.0-beta.17",
    note = "guards the legacy clone workflow; use the dataset clone APIs, which relocate fragment reuse mapping references"
)]
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
    use crate::io::commit::{ManifestNamingScheme, write_manifest_file_to_path};
    use crate::transaction::test_support::sample_manifest;
    use object_store::path::Path;
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

    /// Writes `manifest` with the given index section into `store` and
    /// returns the location the deprecated guard reads it back from.
    async fn write_located(
        store: &ObjectStore,
        manifest: &mut Manifest,
        indices: Vec<IndexMetadata>,
    ) -> ManifestLocation {
        let path = Path::from("_versions/1.manifest");
        let written = write_manifest_file_to_path(store, manifest, Some(indices), &path, None)
            .await
            .unwrap();
        ManifestLocation {
            version: 1,
            path,
            size: Some(written.size as u64),
            naming_scheme: ManifestNamingScheme::V1,
            e_tag: written.e_tag,
            identity: None,
        }
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn ensure_clone_supported_accepts_history_without_tagged_entries() {
        let store = ObjectStore::memory();

        // Without the sticky flag the guard answers without reading anything:
        // the location points at a manifest that was never written.
        let unflagged = sample_manifest();
        let missing = ManifestLocation {
            version: 1,
            path: Path::from("_versions/missing.manifest"),
            size: None,
            naming_scheme: ManifestNamingScheme::V1,
            e_tag: None,
            identity: None,
        };
        ensure_clone_supported(&store, &missing, &unflagged)
            .await
            .unwrap();

        // The sticky flag alone is not a rejection: a v0 entry and a user
        // index carry no mapping references to relocate.
        let mut flagged = sample_manifest();
        flagged.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        flagged.writer_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        let location = write_located(
            &store,
            &mut flagged,
            vec![entry(FRAG_REUSE_INDEX_NAME, 0), entry("user_idx", 1)],
        )
        .await;
        ensure_clone_supported(&store, &location, &flagged)
            .await
            .unwrap();
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn ensure_clone_supported_rejects_tagged_history() {
        let store = ObjectStore::memory();
        let mut flagged = sample_manifest();
        flagged.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        flagged.writer_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        let location =
            write_located(&store, &mut flagged, vec![entry(FRAG_REUSE_INDEX_NAME, 1)]).await;

        let err = ensure_clone_supported(&store, &location, &flagged)
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::NotSupported { .. }),
            "expected NotSupported, got {err:?}"
        );
        assert!(
            err.to_string()
                .contains("Cloning tagged FRI requires row-map reference relocation"),
            "unexpected message: {err}"
        );
    }
}
