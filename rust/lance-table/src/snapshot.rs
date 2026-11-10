// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lightweight, engine-independent references to exact table snapshots.

use bytes::Buf;
use lance_core::{Error, Result};
use lance_io::object_store::uri_to_url;
use object_store::path::Path;
use prost::Message;

use crate::io::commit::ManifestLocation;

#[doc(hidden)]
pub mod pb {
    include!("pb/lance.snapshot.rs");
}

/// A portable snapshot identity whose size does not depend on the manifest size.
///
/// The caller supplies a table incarnation identity that distinguishes tables
/// recreated at the same URI and different storage/catalog authorities. Resolving
/// credentials and checking that authority remain the application's responsibility.
/// This reference neither authenticates a worker nor protects files from cleanup.
///
/// ```
/// # use lance_core::Result;
/// # use lance_table::{snapshot::SnapshotRef, io::commit::ManifestLocation};
/// # fn example(location: &ManifestLocation, digest: [u8; 32]) -> Result<()> {
/// let snapshot = SnapshotRef::try_new(
///     "catalog/table-incarnation", "s3://bucket/table.lance", None, location, digest,
/// )?;
/// assert_eq!(snapshot, SnapshotRef::decode(&snapshot.encode())?);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotRef {
    proto: pb::SnapshotRef,
    snapshot_id: [u8; 32],
}

impl SnapshotRef {
    /// Describe a resolved manifest using a digest of its complete original file.
    ///
    /// `branch = None` identifies the main branch; a named branch must be resolved
    /// to `location.version`. `table_uri` is the absolute root, without a branch
    /// suffix. `table_identity` must include the table incarnation and authority.
    pub fn try_new(
        table_identity: impl Into<String>,
        table_uri: impl AsRef<str>,
        branch: Option<String>,
        location: &ManifestLocation,
        manifest_digest: [u8; 32],
    ) -> Result<Self> {
        Self::try_from(pb::SnapshotRef {
            protocol_version: 1,
            table_identity: table_identity.into(),
            table_uri: canonical_uri(table_uri.as_ref())?,
            branch,
            version: location.version,
            manifest_path: location.path.to_string(),
            manifest_digest: manifest_digest.to_vec(),
            manifest_identity: location.identity.clone(),
        })
    }

    /// Check that a resolved table location names this reference's root.
    ///
    /// Local paths and equivalent file URIs are accepted. A trailing directory
    /// slash is normalized while URI query parameters remain part of the authority.
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_table::snapshot::SnapshotRef;
    /// # fn example(snapshot: &SnapshotRef) -> Result<()> {
    /// snapshot.validate_table_uri(snapshot.table_uri())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate_table_uri(&self, uri: impl AsRef<str>) -> Result<()> {
        let observed = canonical_uri(uri.as_ref())?;
        if observed != self.table_uri() {
            return Err(Error::invalid_input(format!(
                "Snapshot table_uri {} does not match table_uri {}",
                self.table_uri(),
                observed,
            )));
        }
        Ok(())
    }

    /// Serialize identity only; the manifest remains in source storage.
    pub fn encode(&self) -> Vec<u8> {
        self.proto.encode_to_vec()
    }

    /// Decode and validate the transport contract without accessing storage.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let proto = pb::SnapshotRef::decode(bytes).map_err(|error| {
            Error::invalid_input(format!("Invalid snapshot reference: {error}"))
        })?;
        Self::try_from(proto)
    }

    /// Identity fingerprint, independent of worker storage bindings and capabilities.
    pub fn snapshot_id(&self) -> &[u8; 32] {
        &self.snapshot_id
    }
    /// Caller-supplied table incarnation and authority.
    pub fn table_identity(&self) -> &str {
        &self.proto.table_identity
    }
    /// Canonical absolute table root URI.
    pub fn table_uri(&self) -> &str {
        &self.proto.table_uri
    }
    /// Exact branch; `None` identifies the main branch.
    pub fn branch(&self) -> Option<&str> {
        self.proto.branch.as_deref()
    }
    /// Exact version within the resolved branch, never an unresolved latest version.
    pub fn version(&self) -> u64 {
        self.proto.version
    }
    /// Resolved manifest object path in the source object store.
    pub fn manifest_path(&self) -> &str {
        &self.proto.manifest_path
    }
    /// BLAKE3 digest of the complete original manifest file, including all sections.
    pub fn manifest_digest(&self) -> &[u8] {
        &self.proto.manifest_digest
    }
    /// Authoritative commit-store record identity, when the handler supplies one.
    pub fn manifest_identity(&self) -> Option<&str> {
        self.proto.manifest_identity.as_deref()
    }
}

impl TryFrom<pb::SnapshotRef> for SnapshotRef {
    type Error = Error;

    fn try_from(proto: pb::SnapshotRef) -> Result<Self> {
        if proto.protocol_version != 1 {
            return Err(Error::invalid_input(format!(
                "Unsupported snapshot reference protocol version {}",
                proto.protocol_version
            )));
        }
        if proto.table_identity.trim().is_empty() {
            return Err(Error::invalid_input(
                "Snapshot table_identity cannot be empty",
            ));
        }
        if proto.table_uri != canonical_uri(&proto.table_uri)? {
            return Err(Error::invalid_input(format!(
                "Noncanonical snapshot table URI {}",
                proto.table_uri
            )));
        }
        if let Some(branch) = &proto.branch
            && (branch.is_empty()
                || branch == "main"
                || branch.contains("..")
                || branch.contains('\\')
                || branch.ends_with(".lock")
                || branch.split('/').any(|part| {
                    part.is_empty()
                        || !part
                            .chars()
                            .all(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '_'))
                }))
        {
            return Err(Error::invalid_input(format!(
                "Invalid snapshot branch {branch:?}; use None for main"
            )));
        }
        if proto.version == 0 {
            return Err(Error::invalid_input(
                "Snapshot version must be greater than 0, got 0",
            ));
        }
        if proto.manifest_path.trim().is_empty() {
            return Err(Error::invalid_input(
                "Snapshot manifest_path cannot be empty",
            ));
        }
        Path::parse(&proto.manifest_path)?;
        if proto.manifest_digest.len() != 32 {
            return Err(Error::invalid_input(format!(
                "Snapshot manifest_digest has {} bytes, expected 32",
                proto.manifest_digest.len()
            )));
        }
        if proto
            .manifest_identity
            .as_ref()
            .is_some_and(String::is_empty)
        {
            return Err(Error::invalid_input(
                "Snapshot manifest_identity cannot be empty when present",
            ));
        }
        let snapshot_id = *blake3::hash(&proto.encode_to_vec()).as_bytes();
        Ok(Self { proto, snapshot_id })
    }
}

impl From<&SnapshotRef> for pb::SnapshotRef {
    fn from(snapshot: &SnapshotRef) -> Self {
        snapshot.proto.clone()
    }
}

/// Hash a complete original manifest file without joining its storage chunks.
///
/// Include index metadata, inline transactions, dictionaries, length prefixes, and
/// the footer. Do not re-encode a decoded manifest: protobuf map ordering and
/// hydrated dictionaries can change its bytes.
///
/// ```
/// use lance_table::snapshot::digest_manifest;
/// let digest = digest_manifest(b"original manifest file bytes".as_slice());
/// assert_eq!(digest.len(), 32);
/// ```
pub fn digest_manifest(mut bytes: impl Buf) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    while bytes.has_remaining() {
        let chunk = bytes.chunk();
        hasher.update(chunk);
        let length = chunk.len();
        bytes.advance(length);
    }
    *hasher.finalize().as_bytes()
}

fn canonical_uri(uri: &str) -> Result<String> {
    if url::Url::parse(uri).is_err() && !std::path::Path::new(uri).is_absolute() {
        return Err(Error::invalid_input(format!(
            "Snapshot URI must be absolute: {uri}"
        )));
    }
    let mut url = uri_to_url(uri)?;
    let has_username = !url.username().is_empty() && !matches!(url.scheme(), "az" | "abfss");
    if url.fragment().is_some() || has_username || url.password().is_some() {
        return Err(Error::invalid_input(
            "Snapshot URI must not contain credentials or a fragment",
        ));
    }
    // Native directory paths gain a slash in uri_to_url, whereas an existing
    // URI need not have one. Normalize just the root path, preserving queries.
    let path = url.path().trim_end_matches('/').to_string();
    url.set_path(&path);
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::commit::ManifestNamingScheme;
    use lance_io::utils::ChunkedBuf;
    use rstest::rstest;

    fn snapshot() -> SnapshotRef {
        SnapshotRef::try_new(
            "catalog/incarnation",
            "s3://bucket/table",
            None,
            &ManifestLocation {
                version: 7,
                path: Path::from("table/_versions/7.manifest"),
                size: None,
                naming_scheme: ManifestNamingScheme::V1,
                e_tag: None,
                identity: None,
                last_modified: None,
            },
            [42; 32],
        )
        .unwrap()
    }

    #[test]
    fn transport_preserves_snapshot_identity() {
        let snapshot = snapshot();
        let decoded = SnapshotRef::decode(&snapshot.encode()).unwrap();
        assert_eq!(snapshot, decoded);
        assert_eq!(snapshot.snapshot_id(), decoded.snapshot_id());
        assert!(snapshot.encode().len() < 256);
        let mut proto = snapshot.proto.clone();
        proto.table_identity = "catalog/recreated".into();
        assert_ne!(
            snapshot.snapshot_id(),
            SnapshotRef::try_from(proto).unwrap().snapshot_id()
        );
    }

    #[rstest]
    #[case::protocol(|p: &mut pb::SnapshotRef| p.protocol_version = 2, "protocol version 2")]
    #[case::identity(|p: &mut pb::SnapshotRef| p.table_identity.clear(), "table_identity")]
    #[case::uri(|p: &mut pb::SnapshotRef| p.table_uri = "relative/table".into(), "absolute")]
    #[case::credentials(|p: &mut pb::SnapshotRef| p.table_uri = "s3://user:secret@bucket/table".into(), "credentials")]
    #[case::branch(|p: &mut pb::SnapshotRef| p.branch = Some("../other".into()), "branch")]
    #[case::main(|p: &mut pb::SnapshotRef| p.branch = Some("main".into()), "branch")]
    #[case::version(|p: &mut pb::SnapshotRef| p.version = 0, "version")]
    #[case::path(|p: &mut pb::SnapshotRef| p.manifest_path.clear(), "manifest_path")]
    #[case::digest(|p: &mut pb::SnapshotRef| p.manifest_digest.clear(), "manifest_digest")]
    #[case::record(|p: &mut pb::SnapshotRef| p.manifest_identity = Some(String::new()), "manifest_identity")]
    fn rejects_invalid_reference(#[case] change: fn(&mut pb::SnapshotRef), #[case] message: &str) {
        let mut proto = snapshot().proto;
        change(&mut proto);
        let error = SnapshotRef::decode(&proto.encode_to_vec()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }

    #[rstest]
    #[case::storage("s3://bucket/table/", "s3://bucket/table")]
    #[case::local_uri("file:///tmp/table/", "file:///tmp/table")]
    #[case::query(
        "s3+ddb://bucket/table/?ddbTableName=commits",
        "s3+ddb://bucket/table?ddbTableName=commits"
    )]
    #[case::escaped("s3://bucket/a%20table/", "s3://bucket/a%20table")]
    fn table_root_normalization(#[case] uri: &str, #[case] expected: &str) {
        assert_eq!(canonical_uri(uri).unwrap(), expected);
        let mut proto = snapshot().proto;
        proto.table_uri = expected.into();
        let snapshot = SnapshotRef::try_from(proto).unwrap();
        snapshot.validate_table_uri(uri).unwrap();
        let error = snapshot.validate_table_uri("s3://other/table").unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error.to_string().contains("does not match table_uri"),
            "{error}"
        );
    }

    #[test]
    fn digest_is_independent_of_chunk_boundaries() {
        let mut chunks = ChunkedBuf::default();
        chunks.push(bytes::Bytes::from_static(b"original "));
        chunks.push(bytes::Bytes::from_static(b"manifest"));
        assert_eq!(
            digest_manifest(chunks),
            digest_manifest(b"original manifest".as_slice())
        );
    }
}
