// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The keep/delete decision for a file found by listing a managed subtree.
//!
//! The decision has two steps. [`ListedFile::classify`] maps a dataset-relative
//! path to its accounting kind and the key manifests would reference it by.
//! [`Verdict::decide`] then applies one rule to every kind, so the safety
//! invariant (never delete a file a retained manifest references, and never
//! delete an unproven file that may belong to an in-progress write) lives in a
//! single place.

use object_store::path::Path;

use super::{CleanupFileKind as Kind, ReferencedFiles, staging_manifest_version};
use crate::dataset::TRANSACTIONS_DIR;

use FileReference as Ref;

/// The key a manifest would reference a listed file by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum FileReference {
    /// Leftovers of a failed write that no manifest can name.
    Unreferenceable,
    /// The commit handler, not manifest contents, decides whether it is live.
    StagingManifest {
        version: u64,
    },
    DataFile(Path),
    /// Sidecars live and die with their parent data file.
    BlobSidecar {
        path: Path,
        data_file: Path,
    },
    ManagedBlob(Path),
    DeletionFile(Path),
    TransactionFile(Path),
    IndexUuid(String),
    RowMapId(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ListedFile {
    pub kind: Kind,
    pub reference: FileReference,
}

impl ListedFile {
    /// Returns `None` for paths outside the layouts cleanup manages; those are
    /// always kept.
    pub(super) fn classify(relative_path: &Path) -> Option<Self> {
        let listed = |kind, reference| Some(Self { kind, reference });

        if relative_path.as_ref().starts_with("_versions/.tmp") {
            return listed(Kind::TemporaryManifest, Ref::Unreferenceable);
        }
        if let Some(version) = staging_manifest_version(relative_path) {
            return listed(Kind::TemporaryManifest, Ref::StagingManifest { version });
        }

        let mut parts = relative_path.parts();
        let top_dir = parts.next()?;
        let top_dir = top_dir.as_ref();
        // Row maps and indices are referenced by the id of their directory
        // (`_fri/<map_id>/...`, `_indices/<uuid>/...`), whatever the file.
        if top_dir == "_fri" {
            return listed(
                Kind::Index,
                Ref::RowMapId(parts.next()?.as_ref().to_string()),
            );
        }
        if top_dir == "_indices" {
            return listed(
                Kind::Index,
                Ref::IndexUuid(parts.next()?.as_ref().to_string()),
            );
        }

        match relative_path.extension() {
            Some("lance") if top_dir == "data" => {
                listed(Kind::Data, Ref::DataFile(relative_path.clone()))
            }
            Some("blob") => Self::classify_blob(relative_path.clone(), top_dir),
            Some("arrow" | "bin") if top_dir == "_deletions" => {
                listed(Kind::Deletion, Ref::DeletionFile(relative_path.clone()))
            }
            Some("txn") if top_dir == TRANSACTIONS_DIR => listed(
                Kind::Transaction,
                Ref::TransactionFile(relative_path.clone()),
            ),
            Some("lance" | "arrow" | "bin" | "txn" | "manifest") => None,
            // Local writers stage files in their destination directory before
            // renaming them, even inside a referenced blob sidecar directory.
            _ if top_dir == "data"
                && relative_path
                    .filename()
                    .is_some_and(|filename| filename.starts_with(".tmp")) =>
            {
                listed(Kind::Data, Ref::Unreferenceable)
            }
            _ => None,
        }
    }

    fn classify_blob(path: Path, top_dir: &str) -> Option<Self> {
        let reference = match top_dir {
            "_blobs" => Ref::ManagedBlob(path),
            "data" => {
                // Blob v2 sidecars are keyed by the data file stem:
                //   data/{data_file_key}/{obfuscated_blob_id:032b}.blob
                // Any other layout under `data/` is left alone.
                let parts = path.parts().collect::<Vec<_>>();
                let [_, data_file_key, _] = parts.as_slice() else {
                    return None;
                };
                let data_file =
                    Path::parse(format!("data/{}.lance", data_file_key.as_ref())).ok()?;
                Ref::BlobSidecar { path, data_file }
            }
            _ => return None,
        };
        Some(Self {
            kind: Kind::Data,
            reference,
        })
    }
}

impl ReferencedFiles {
    pub(super) fn contains(&self, reference: &FileReference) -> bool {
        match reference {
            Ref::Unreferenceable | Ref::StagingManifest { .. } => false,
            Ref::DataFile(path) => self.data_paths.contains(path),
            Ref::BlobSidecar { path, data_file } => {
                self.data_paths.contains(data_file) || self.managed_blob_paths.contains(path)
            }
            Ref::ManagedBlob(path) => self.managed_blob_paths.contains(path),
            Ref::DeletionFile(path) => self.delete_paths.contains(path),
            Ref::TransactionFile(path) => self.tx_paths.contains(path),
            Ref::IndexUuid(uuid) => self.index_uuids.contains(uuid),
            Ref::RowMapId(map_id) => self.frag_reuse_map_ids.contains(map_id),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    Keep,
    /// `unverified` is set when no manifest proves the file belonged to a
    /// finished commit, so it is deleted only on age or `delete_unverified`.
    Delete {
        unverified: bool,
    },
}

impl Verdict {
    /// * `is_referenced`: some working-set manifest references the file.
    /// * `is_verified`: some manifest, possibly expiring, references the file,
    ///   so it is not part of an in-progress write.
    /// * `maybe_in_progress`: the file is young and `delete_unverified` is off.
    pub(super) fn decide(is_referenced: bool, is_verified: bool, maybe_in_progress: bool) -> Self {
        if is_referenced {
            Self::Keep
        } else if is_verified {
            Self::Delete { unverified: false }
        } else if maybe_in_progress {
            Self::Keep
        } else {
            Self::Delete { unverified: true }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const UUID: &str = "0b5e2a7c-8a4f-4b8e-9d3a-2f1c6e7d8a90";

    fn path(path: &str) -> Path {
        Path::parse(path).unwrap()
    }

    fn listed(kind: Kind, reference: FileReference) -> Option<ListedFile> {
        Some(ListedFile { kind, reference })
    }

    #[rstest]
    #[case::temporary_manifest(
        "_versions/.tmp/orphan",
        listed(Kind::TemporaryManifest, Ref::Unreferenceable)
    )]
    #[case::staging_manifest(
        &format!("_versions/3.manifest-{UUID}"),
        listed(Kind::TemporaryManifest, Ref::StagingManifest { version: 3 })
    )]
    #[case::detached_staging_manifest(
        &format!("_versions/d{}.manifest-{UUID}", lance_table::format::DETACHED_VERSION_MASK | 7),
        listed(
            Kind::TemporaryManifest,
            Ref::StagingManifest {
                version: lance_table::format::DETACHED_VERSION_MASK | 7
            },
        )
    )]
    #[case::manifest("_versions/3.manifest", None)]
    #[case::manifest_with_bad_suffix("_versions/3.manifest-not-a-uuid", None)]
    #[case::data_file(
        "data/abc.lance",
        listed(Kind::Data, Ref::DataFile(path("data/abc.lance")))
    )]
    #[case::lance_file_outside_data("_blobs/abc.lance", None)]
    #[case::blob_sidecar(
        "data/abc/0001.blob",
        listed(
            Kind::Data,
            Ref::BlobSidecar {
                path: path("data/abc/0001.blob"),
                data_file: path("data/abc.lance"),
            },
        )
    )]
    #[case::blob_directly_in_data("data/abc.blob", None)]
    #[case::blob_nested_too_deep("data/abc/nested/0001.blob", None)]
    #[case::managed_blob(
        "_blobs/a/b/0001.blob",
        listed(Kind::Data, Ref::ManagedBlob(path("_blobs/a/b/0001.blob")))
    )]
    #[case::blob_outside_managed_dirs("images/clip.blob", None)]
    #[case::deletion_arrow(
        "_deletions/0-1-2.arrow",
        listed(Kind::Deletion, Ref::DeletionFile(path("_deletions/0-1-2.arrow")))
    )]
    #[case::deletion_bin(
        "_deletions/0-1-2.bin",
        listed(Kind::Deletion, Ref::DeletionFile(path("_deletions/0-1-2.bin")))
    )]
    #[case::bin_outside_deletions("data/abc.bin", None)]
    #[case::transaction(
        "_transactions/1-abc.txn",
        listed(
            Kind::Transaction,
            Ref::TransactionFile(path("_transactions/1-abc.txn")),
        )
    )]
    #[case::txn_outside_transactions("data/1-abc.txn", None)]
    #[case::index_file(
        &format!("_indices/{UUID}/index.idx"),
        listed(Kind::Index, Ref::IndexUuid(UUID.to_string()))
    )]
    // Index directories hold `.lance` files that no fragment references.
    #[case::index_lance_file(
        &format!("_indices/{UUID}/auxiliary.lance"),
        listed(Kind::Index, Ref::IndexUuid(UUID.to_string()))
    )]
    #[case::nested_index_file(
        &format!("_indices/{UUID}/partial_x/index.idx"),
        listed(Kind::Index, Ref::IndexUuid(UUID.to_string()))
    )]
    #[case::bare_indices_dir("_indices", None)]
    #[case::row_map(
        "_fri/map-1/part-0.lance",
        listed(Kind::Index, Ref::RowMapId("map-1".to_string()))
    )]
    #[case::bare_row_map_dir("_fri", None)]
    #[case::local_staging_in_data("data/abc/.tmp123", listed(Kind::Data, Ref::Unreferenceable))]
    #[case::local_staging_with_extension_in_data(
        "data/.tmp123.partial",
        listed(Kind::Data, Ref::Unreferenceable)
    )]
    #[case::local_staging_outside_data("_blobs/.tmp123", None)]
    #[case::unknown_data_file("data/notes.txt", None)]
    fn classify(#[case] relative_path: &str, #[case] expected: Option<ListedFile>) {
        assert_eq!(ListedFile::classify(&path(relative_path)), expected);
    }

    #[rstest]
    #[case::referenced_young(true, false, true, Verdict::Keep)]
    #[case::referenced_old(true, false, false, Verdict::Keep)]
    #[case::referenced_and_verified(true, true, false, Verdict::Keep)]
    #[case::verified_young(false, true, true, Verdict::Delete { unverified: false })]
    // Regression: data, index, deletion and row-map files used to report a
    // verified old file as unverified.
    #[case::verified_old(false, true, false, Verdict::Delete { unverified: false })]
    #[case::unproven_young(false, false, true, Verdict::Keep)]
    #[case::unproven_old(false, false, false, Verdict::Delete { unverified: true })]
    fn decide(
        #[case] is_referenced: bool,
        #[case] is_verified: bool,
        #[case] maybe_in_progress: bool,
        #[case] expected: Verdict,
    ) {
        assert_eq!(
            Verdict::decide(is_referenced, is_verified, maybe_in_progress),
            expected
        );
    }

    /// Manifests that name one file of every kind.
    fn manifests_naming_one_of_each() -> ReferencedFiles {
        ReferencedFiles {
            data_paths: [path("data/abc.lance")].into(),
            managed_blob_paths: [path("_blobs/a/0001.blob"), path("data/old/0001.blob")].into(),
            delete_paths: [path("_deletions/0-1-2.arrow")].into(),
            tx_paths: [path("_transactions/1-abc.txn")].into(),
            index_uuids: [UUID.to_string()].into(),
            frag_reuse_map_ids: ["map-1".to_string()].into(),
        }
    }

    #[rstest]
    #[case::data_file("data/abc.lance")]
    #[case::sidecar_by_parent_data_file("data/abc/0001.blob")]
    #[case::sidecar_by_own_path("data/old/0001.blob")]
    #[case::managed_blob("_blobs/a/0001.blob")]
    #[case::deletion_file("_deletions/0-1-2.arrow")]
    #[case::transaction_file("_transactions/1-abc.txn")]
    #[case::index_file(&format!("_indices/{UUID}/index.idx"))]
    #[case::row_map_file("_fri/map-1/part-0.lance")]
    fn manifests_reference_files_by_their_key(#[case] relative_path: &str) {
        let file = ListedFile::classify(&path(relative_path)).unwrap();
        assert!(manifests_naming_one_of_each().contains(&file.reference));
        assert!(!ReferencedFiles::default().contains(&file.reference));
    }

    #[rstest]
    #[case::data_file("data/other.lance")]
    #[case::sidecar("data/other/0001.blob")]
    #[case::managed_blob("_blobs/a/0002.blob")]
    #[case::deletion_file("_deletions/0-1-3.arrow")]
    #[case::transaction_file("_transactions/2-abc.txn")]
    #[case::index_file("_indices/other-uuid/index.idx")]
    #[case::row_map_file("_fri/map-2/part-0.lance")]
    #[case::local_staging("data/abc/.tmp123")]
    #[case::temporary_manifest("_versions/.tmp/orphan")]
    #[case::staging_manifest(&format!("_versions/3.manifest-{UUID}"))]
    fn manifests_do_not_reference_other_keys(#[case] relative_path: &str) {
        let file = ListedFile::classify(&path(relative_path)).unwrap();
        assert!(!manifests_naming_one_of_each().contains(&file.reference));
    }
}
