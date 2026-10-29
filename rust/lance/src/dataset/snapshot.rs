// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Storage identity shared by ordinary queries and transported blob references.

use bytes::Buf;
use lance_core::{Error, Result};
use lance_io::object_store::uri_to_url;
use lance_table::{
    format::{Manifest, pb as table_pb},
    io::manifest::read_manifest_bytes,
};
use object_store::path::Path;
use prost::Message;

use crate::Dataset;
use crate::dataset::refs::check_valid_branch;

pub(crate) mod pb {
    include!(concat!(env!("OUT_DIR"), "/lance.dataset.retention.rs"));
}

/// The exact table incarnation, branch, version, and original manifest identity.
///
/// It contains no credentials or reader-specific capabilities. Holding or decoding
/// an identity does not protect its files; use
/// [`SnapshotRetention`](crate::dataset::retention::SnapshotRetention).
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::snapshot::SnapshotIdentity;
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let snapshot = SnapshotIdentity::from_dataset(dataset, "catalog/table-incarnation").await?;
/// assert_eq!(snapshot, SnapshotIdentity::decode(&snapshot.encode())?);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotIdentity {
    proto: pb::SnapshotIdentity,
    snapshot_id: [u8; 32],
}

impl SnapshotIdentity {
    /// Capture the loaded snapshot's original manifest without reading data payloads.
    /// `table_identity` must distinguish table incarnations that can reuse a URI.
    pub async fn from_dataset(
        dataset: &Dataset,
        table_identity: impl Into<String>,
    ) -> Result<Self> {
        let location = dataset.manifest_location();
        let mut manifest =
            read_manifest_bytes(&dataset.object_store, &location.path, location.size).await?;
        let manifest = manifest.copy_to_bytes(manifest.remaining());
        let snapshot = Self::from_proto(pb::SnapshotIdentity {
            protocol_version: 1,
            table_identity: table_identity.into(),
            table_uri: canonical_uri(&dataset.refs.root()?.uri)?,
            branch: dataset.manifest().branch.clone(),
            version: dataset.manifest().version,
            manifest_path: location.path.to_string(),
            manifest,
            manifest_identity: location.identity.clone(),
        })?;
        snapshot.validate_dataset(dataset)?;
        Ok(snapshot)
    }

    pub(crate) fn from_proto(proto: pb::SnapshotIdentity) -> Result<Self> {
        if proto.protocol_version != 1 {
            return Err(Error::invalid_input(format!(
                "Unsupported snapshot identity protocol version {}",
                proto.protocol_version
            )));
        }
        for (name, value) in [
            ("table_identity", &proto.table_identity),
            ("manifest_path", &proto.manifest_path),
        ] {
            if value.trim().is_empty() {
                return Err(Error::invalid_input(format!(
                    "Snapshot {name} must not be empty, got {value:?}"
                )));
            }
        }
        if proto.table_uri != canonical_uri(&proto.table_uri)? {
            return Err(Error::invalid_input(format!(
                "Noncanonical table URI {}",
                proto.table_uri
            )));
        }
        Path::parse(&proto.manifest_path)?;
        if let Some(branch) = &proto.branch {
            check_valid_branch(branch)?;
        }
        if proto.version == 0 {
            return Err(Error::invalid_input(
                "Snapshot version must be positive, got 0",
            ));
        }
        if proto
            .manifest_identity
            .as_ref()
            .is_some_and(String::is_empty)
        {
            return Err(Error::invalid_input(
                "Snapshot manifest_identity must not be empty when present",
            ));
        }
        let manifest = table_pb::Manifest::decode(proto.manifest.clone()).map_err(|error| {
            Error::invalid_input(format!("Invalid snapshot manifest protobuf: {error}"))
        })?;
        if manifest.version != proto.version || manifest.branch != proto.branch {
            return Err(Error::invalid_input(format!(
                "Snapshot {:?}/{} disagrees with its original manifest {:?}/{}",
                proto.branch, proto.version, manifest.branch, manifest.version
            )));
        }
        let snapshot_id = *blake3::hash(&proto.encode_to_vec()).as_bytes();
        Ok(Self { proto, snapshot_id })
    }

    /// Encode the identity for a coordinator recovery record or another process.
    pub fn encode(&self) -> Vec<u8> {
        self.proto.encode_to_vec()
    }

    /// Validate a transported identity without acquiring protection.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let proto = pb::SnapshotIdentity::decode(bytes)
            .map_err(|error| Error::invalid_input(format!("Invalid snapshot identity: {error}")))?;
        Self::from_proto(proto)
    }

    /// Fingerprint independent of blob capabilities and worker storage binding keys.
    pub fn snapshot_id(&self) -> &[u8; 32] {
        &self.snapshot_id
    }
    /// Caller-supplied table incarnation and authority.
    pub fn table_identity(&self) -> &str {
        &self.proto.table_identity
    }
    /// Canonical table root URI.
    pub fn table_uri(&self) -> &str {
        &self.proto.table_uri
    }
    /// Exact branch; `None` identifies the main branch.
    pub fn branch(&self) -> Option<&str> {
        self.proto.branch.as_deref()
    }
    /// Exact snapshot version, never an unresolved branch head.
    pub fn version(&self) -> u64 {
        self.proto.version
    }
    /// Original manifest protobuf bytes, including registered bases.
    pub fn serialized_manifest(&self) -> &[u8] {
        &self.proto.manifest
    }

    pub(crate) fn validate_dataset(&self, dataset: &Dataset) -> Result<()> {
        let location = dataset.manifest_location();
        let mut observed =
            Manifest::try_from(table_pb::Manifest::decode(self.proto.manifest.clone())?)?;
        observed.detach_sparse_inline_row_ids(self.proto.manifest.len());
        if self.proto.table_uri != canonical_uri(&dataset.refs.root()?.uri)?
            || self.proto.manifest_path != location.path.as_ref()
            || self.proto.manifest_identity != location.identity
            || !matches_loaded_manifest(&observed, dataset.manifest())
        {
            return Err(Error::invalid_input(format!(
                "Snapshot {:?}/{} at {} does not match dataset {:?}/{} at {}",
                self.branch(),
                self.version(),
                self.table_uri(),
                dataset.manifest().branch,
                dataset.version().version,
                dataset.uri()
            )));
        }
        Ok(())
    }
}

pub(crate) fn matches_loaded_manifest(observed: &Manifest, loaded: &Manifest) -> bool {
    // Loaded schemas can contain hydrated dictionaries absent from persisted metadata.
    let mut expected = loaded.clone();
    let mut fields = expected.schema.fields.iter_mut().collect::<Vec<_>>();
    while let Some(field) = fields.pop() {
        if let Some(dictionary) = &mut field.dictionary {
            dictionary.values = None;
        }
        fields.extend(&mut field.children);
    }
    observed == &expected
}

pub(crate) fn canonical_uri(uri: &str) -> Result<String> {
    if url::Url::parse(uri).is_err() && !std::path::Path::new(uri).is_absolute() {
        return Err(Error::invalid_input(format!(
            "Snapshot URI must be absolute: {uri}"
        )));
    }
    let url = uri_to_url(uri)?;
    let has_username = !url.username().is_empty() && !matches!(url.scheme(), "az" | "abfss");
    if url.fragment().is_some() || has_username || url.password().is_some() {
        return Err(Error::invalid_input(
            "Snapshot URI must not contain credentials or a fragment",
        ));
    }
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[cfg_attr(coverage, coverage(off))]
    fn identity() -> SnapshotIdentity {
        SnapshotIdentity::from_proto(pb::SnapshotIdentity {
            protocol_version: 1,
            table_identity: "catalog/table-incarnation".to_string(),
            table_uri: "s3://base/table.lance".to_string(),
            version: 1,
            manifest_path: "table.lance/_versions/1.manifest".to_string(),
            manifest: table_pb::Manifest {
                version: 1,
                ..Default::default()
            }
            .encode_to_vec()
            .into(),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn transport_preserves_identity_and_table_incarnation() {
        let original = identity();
        let decoded = SnapshotIdentity::decode(&original.encode()).unwrap();
        assert_eq!(original, decoded);
        assert_eq!(original.snapshot_id(), decoded.snapshot_id());
        let mut recreated = original.proto.clone();
        recreated.table_identity = "catalog/recreated-table".to_string();
        assert_ne!(
            original.snapshot_id(),
            SnapshotIdentity::from_proto(recreated)
                .unwrap()
                .snapshot_id()
        );
        let mut committed = original.proto.clone();
        committed.manifest_identity = Some("another-commit".to_string());
        assert_ne!(
            original.snapshot_id(),
            SnapshotIdentity::from_proto(committed)
                .unwrap()
                .snapshot_id()
        );
    }

    #[rstest]
    #[case::protocol(|p: &mut pb::SnapshotIdentity| p.protocol_version = 2, "protocol version 2")]
    #[case::empty_table(|p: &mut pb::SnapshotIdentity| p.table_identity.clear(), "table_identity")]
    #[case::empty_manifest_path(|p: &mut pb::SnapshotIdentity| p.manifest_path.clear(), "manifest_path")]
    #[case::relative_uri(|p: &mut pb::SnapshotIdentity| p.table_uri = "relative/table".to_string(), "absolute")]
    #[case::credentials(|p: &mut pb::SnapshotIdentity| p.table_uri = "s3://user:secret@base/table.lance".to_string(), "credentials")]
    #[case::zero_version(|p: &mut pb::SnapshotIdentity| p.version = 0, "got 0")]
    #[case::wrong_version(|p: &mut pb::SnapshotIdentity| p.version = 2, "disagrees")]
    #[case::wrong_branch(|p: &mut pb::SnapshotIdentity| p.branch = Some("other".to_string()), "disagrees")]
    #[case::empty_commit_identity(|p: &mut pb::SnapshotIdentity| p.manifest_identity = Some(String::new()), "manifest_identity")]
    #[case::corrupt_manifest(|p: &mut pb::SnapshotIdentity| p.manifest = vec![255].into(), "manifest protobuf")]
    fn reject_invalid_identity(
        #[case] modify: fn(&mut pb::SnapshotIdentity),
        #[case] message: &str,
    ) {
        let mut proto = identity().proto;
        modify(&mut proto);
        let error = SnapshotIdentity::decode(&proto.encode_to_vec()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }
}
