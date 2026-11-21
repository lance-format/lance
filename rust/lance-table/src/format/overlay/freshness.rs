// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-overlay exceptions carried by ranked index segments.

use std::borrow::Cow;
use std::sync::Arc;

use lance_core::{Error, Result};
use prost::Message;
use prost::encoding::{DecodeContext, WireType, decode_key, skip_field};
use prost_types::Any;

use super::DataOverlayFile;
use crate::format::IndexMetadata;
use crate::format::pb::{
    IndexOverlayFreshness, IndexOverlayFreshnessList, IndexedDataOverlay, RankedIndexOverlayDetails,
};

/// Whether the built-in index format uses the shared overlay field.
pub fn has_ranked_details(details: &Any) -> bool {
    let Some((_, name)) = details.type_url.rsplit_once('/') else {
        return false;
    };
    [
        "lance.index.pb.VectorIndexDetails",
        "lance.index.VectorIndexDetails",
        "lance.table.InvertedIndexDetails",
        "lance.index.pb.InvertedIndexDetails",
    ]
    .iter()
    .any(|ranked| name.eq_ignore_ascii_case(ranked))
}

/// The original configuration bytes, including fields unknown to this runtime.
/// Overlay facts describe one segment and do not participate in configuration equality.
pub fn index_configuration(details: &Any) -> Result<Cow<'_, [u8]>> {
    let bytes = details.value.as_slice();
    if !has_ranked_details(details) {
        return Ok(Cow::Borrowed(bytes));
    }
    let mut remaining = bytes;
    let mut config = None;
    let mut keep_from = 0;
    while !remaining.is_empty() {
        let start = bytes.len() - remaining.len();
        let (tag, wire_type) = decode_key(&mut remaining)?;
        if tag == 32 && wire_type != WireType::LengthDelimited {
            return Err(Error::index(format!(
                "Invalid overlay field wire type {wire_type:?} in {}",
                details.type_url
            )));
        }
        skip_field(wire_type, tag, &mut remaining, DecodeContext::default())?;
        // RankedIndexOverlayDetails projects field 32 from both ranked index types.
        if tag == 32 {
            if remaining.is_empty() && config.is_none() {
                return Ok(Cow::Borrowed(&bytes[..start]));
            }
            config
                .get_or_insert_with(Vec::new)
                .extend_from_slice(&bytes[keep_from..start]);
            keep_from = bytes.len() - remaining.len();
        }
    }
    Ok(match config {
        Some(mut config) => {
            config.extend_from_slice(&bytes[keep_from..]);
            Cow::Owned(config)
        }
        None => Cow::Borrowed(bytes),
    })
}

/// Whether the ranked details carry exceptions beyond the segment's build version.
pub fn has_overlay_freshness(details: &Any) -> Result<bool> {
    Ok(index_configuration(details)?.len() != details.value.len())
}

/// Share the original details when possible, copying only configuration bytes
/// when a model or a newly built segment needs to shed its source's overlay facts.
pub fn configuration_details(details: &Arc<Any>) -> Result<Arc<Any>> {
    let config = index_configuration(details)?;
    if config.len() == details.value.len() {
        return Ok(Arc::clone(details));
    }
    Ok(Arc::new(Any {
        type_url: details.type_url.clone(),
        value: config.into_owned(),
    }))
}

/// Replace a ranked segment's exceptions while preserving the original encoding of its configuration.
pub fn set_overlay_freshness(
    details: &mut Any,
    entries: Vec<IndexOverlayFreshness>,
    validation_version: Option<u64>,
) -> Result<()> {
    if !has_ranked_details(details) {
        if !entries.is_empty() {
            return Err(Error::index(format!(
                "Cannot attach ranked overlay metadata to {}",
                details.type_url
            )));
        }
        return Ok(());
    }
    let config = index_configuration(details)?;
    if config.len() == details.value.len() && entries.is_empty() {
        return Ok(());
    }
    match config {
        Cow::Borrowed(prefix) => details.value.truncate(prefix.len()),
        Cow::Owned(bytes) => details.value = bytes,
    }
    if !entries.is_empty() {
        RankedIndexOverlayDetails {
            overlay_freshness: Some(IndexOverlayFreshnessList {
                entries,
                validation_version,
            }),
        }
        .encode(&mut details.value)?;
    }
    Ok(())
}

/// Sorted by physical overlay identity, with base IDs resolved against the segment.
#[derive(Default)]
pub struct OverlayFreshness {
    entries: Vec<IndexOverlayFreshness>,
    validation_version: Option<u64>,
}

fn identity(entry: &IndexedDataOverlay) -> (u64, &str, Option<u32>) {
    (entry.committed_version, &entry.path, entry.base_id)
}

impl OverlayFreshness {
    /// Read only the overlay field from a ranked index's details.
    pub fn read(segment: &IndexMetadata) -> Result<Self> {
        let Some(details) = segment.index_details.as_deref() else {
            return Ok(Self::default());
        };
        Self::from_details(segment.uuid, details, segment.base_id)
    }

    /// Decode an index's physical overlay facts while it is still staged.
    pub fn from_details(uuid: uuid::Uuid, details: &Any, base_id: Option<u32>) -> Result<Self> {
        if !has_ranked_details(details) {
            return Ok(Self::default());
        }
        let Some(list) =
            RankedIndexOverlayDetails::decode(details.value.as_slice())?.overlay_freshness
        else {
            return Ok(Self::default());
        };
        let mut entries = list.entries;
        entries.sort_unstable_by_key(|entry| entry.fragment_id);
        if entries
            .windows(2)
            .any(|pair| pair[0].fragment_id == pair[1].fragment_id)
        {
            return Err(Error::index(format!(
                "Duplicate fragment in overlay metadata for index segment {uuid}"
            )));
        }
        for entry in &mut entries {
            for overlay in &mut entry.overlays {
                overlay.base_id = overlay.base_id.or(base_id);
            }
            entry
                .overlays
                .sort_unstable_by(|a, b| identity(a).cmp(&identity(b)));
            if entry.overlays.is_empty()
                || entry
                    .overlays
                    .windows(2)
                    .any(|pair| identity(&pair[0]) == identity(&pair[1]))
            {
                return Err(Error::index(format!(
                    "Empty or duplicate overlay list for fragment {} in index segment {uuid}",
                    entry.fragment_id
                )));
            }
            if base_id.is_some() {
                entry.source_versions.clear();
            }
        }
        Ok(Self {
            entries,
            validation_version: list.validation_version.filter(|_| base_id.is_none()),
        })
    }

    /// Whether the ordinary segment-wide version comparison applies to every overlay.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The local snapshot that endorsed coverage inherited from another dataset.
    pub fn validation_version(&self) -> Option<u64> {
        self.validation_version
    }

    /// The recorded overlays and source versions, grouped by fragment.
    pub fn entries(&self) -> &[IndexOverlayFreshness] {
        &self.entries
    }

    /// The incorporated files and source snapshots for one fragment.
    pub fn for_fragment(&self, fragment: u32) -> Option<&IndexOverlayFreshness> {
        self.entries
            .binary_search_by_key(&fragment, |entry| entry.fragment_id)
            .ok()
            .map(|i| &self.entries[i])
    }

    /// The snapshots containing this fragment's indexed values. An empty
    /// slice means its rows cannot be verified against this dataset's history.
    pub fn source_versions<'a>(&'a self, fragment: u32, default: &'a u64) -> &'a [u64] {
        self.for_fragment(fragment).map_or_else(
            || std::slice::from_ref(default),
            |entry| entry.source_versions.as_slice(),
        )
    }

    /// Resolve relative references before writing the segment into the local index directory.
    /// A deep clone also copies every referenced overlay into the local data directory.
    pub fn relocate_to_local(segment: &mut IndexMetadata, copy_overlays: bool) -> Result<()> {
        let Self {
            mut entries,
            validation_version,
        } = Self::read(segment)?;
        if entries.is_empty() {
            return Ok(());
        }
        if copy_overlays {
            for entry in &mut entries {
                for overlay in &mut entry.overlays {
                    overlay.base_id = None;
                }
                entry.overlays.dedup();
                entry.source_versions.clear();
            }
        }
        if let Some(details) = &mut segment.index_details {
            set_overlay_freshness(
                Arc::make_mut(details),
                entries,
                validation_version.filter(|_| !copy_overlays),
            )?;
        }
        Ok(())
    }

    /// Classify one live overlay in the same base-path space as the segment.
    pub fn is_stale(&self, fragment: u32, overlay: &DataOverlayFile, dataset_version: u64) -> bool {
        let key = (
            overlay.committed_version,
            overlay.data_file.path.as_str(),
            overlay.data_file.base_id,
        );
        overlay.committed_version > dataset_version
            && self.for_fragment(fragment).is_none_or(|entry| {
                entry
                    .overlays
                    .binary_search_by(|file| identity(file).cmp(&key))
                    .is_err()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::format::overlay::OverlayCoverage;
    use roaring::RoaringBitmap;

    fn fragment() -> IndexOverlayFreshness {
        IndexOverlayFreshness {
            fragment_id: 1,
            source_versions: vec![9],
            overlays: vec![IndexedDataOverlay {
                committed_version: 7,
                path: "patch.lance".into(),
                base_id: None,
            }],
        }
    }

    #[test]
    fn config_preserves_unknown_fields_and_skips_the_overlay_block() {
        let mut details = Any {
            type_url: "/lance.table.InvertedIndexDetails".into(),
            // Includes an unknown varint field with a noncanonical byte encoding.
            value: vec![0x08, 1, 0xf8, 0x02, 0x81, 0],
        };
        let original = details.value.clone();
        set_overlay_freshness(&mut details, Vec::new(), None).unwrap();
        assert_eq!(details.value, original);
        set_overlay_freshness(&mut details, vec![fragment()], Some(9)).unwrap();
        assert!(
            matches!(index_configuration(&details).unwrap(), Cow::Borrowed(bytes) if bytes == original)
        );
        details.value.extend([0x10, 2]);
        let mut expected = original;
        expected.extend([0x10, 2]);
        assert_eq!(index_configuration(&details).unwrap().as_ref(), expected);
        set_overlay_freshness(&mut details, Vec::new(), None).unwrap();
        assert_eq!(details.value, expected);

        details.type_url = "/third.party.InvertedIndexDetails".into();
        assert!(set_overlay_freshness(&mut details, vec![fragment()], Some(9)).is_err());
        assert_eq!(details.value, expected);
    }

    #[test]
    fn overlay_identity_includes_fragment_version_path_and_resolved_base() {
        let mut details = Any {
            type_url: "/lance.index.pb.VectorIndexDetails".into(),
            value: Vec::new(),
        };
        set_overlay_freshness(&mut details, vec![fragment()], Some(9)).unwrap();
        let id = uuid::Uuid::new_v4();
        let local = OverlayFreshness::from_details(id, &details, None).unwrap();
        let external = OverlayFreshness::from_details(id, &details, Some(3)).unwrap();
        assert_eq!(local.source_versions(1, &1), [9]);
        assert!(external.source_versions(1, &1).is_empty());
        assert_eq!(local.validation_version(), Some(9));
        assert_eq!(external.validation_version(), None);
        let mut overlay = DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("patch.lance", vec![0], None),
            coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0])),
            committed_version: 7,
        };
        assert!(!local.is_stale(1, &overlay, 1));
        assert!(external.is_stale(1, &overlay, 1));
        overlay.data_file.base_id = Some(3);
        assert!(!external.is_stale(1, &overlay, 1));
        assert!(external.is_stale(0, &overlay, 1));
        overlay.committed_version += 1;
        assert!(external.is_stale(1, &overlay, 1));
        overlay.committed_version -= 1;
        overlay.data_file.path = "other.lance".into();
        assert!(external.is_stale(1, &overlay, 1));
        assert!(!external.is_stale(1, &overlay, 7));
    }
}
