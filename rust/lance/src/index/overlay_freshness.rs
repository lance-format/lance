// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Overlay visibility for ranked indices whose entries come from different snapshots.

use std::sync::Arc;

use lance_core::Result;
use lance_select::{RowAddrTreeMap, RowSetOps};
use lance_table::feature_flags::FLAG_UNSTABLE_DATA_OVERLAY_FILES;
use lance_table::format::overlay::freshness::{
    OverlayFreshness, configuration_details, set_overlay_freshness,
};
use lance_table::format::overlay::staleness::{field_affects_index, stale_overlay_offsets};
use lance_table::format::pb::{IndexOverlayFreshness, IndexedDataOverlay};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;

use crate::Dataset;

/// Incorporated overlays and rows to refresh before publishing a merged segment.
#[derive(Default)]
pub struct MergeOverlayFreshness {
    indexed: Vec<IndexOverlayFreshness>,
    validation_version: Option<u64>,
    /// Rows that must be re-indexed if address remapping advances the output version.
    pub(super) refresh: RowAddrTreeMap,
}

impl MergeOverlayFreshness {
    /// Write the remaining facts after any rows requiring a refresh have been rebuilt.
    pub async fn apply(self, dataset: &Dataset, output: &mut IndexMetadata) -> Result<()> {
        if !self.refresh.is_empty() {
            return Err(lance_core::Error::internal(
                "A ranked index merge would publish unrefreshed overlay rows",
            ));
        }
        // Older readers treat nonempty vector details as a complete model description.
        // Infer the real metric before adding the first overlay field to an old segment.
        if !self.indexed.is_empty()
            && super::vector::details::needs_vector_details_inference(output, dataset.schema())
        {
            output.index_details = Some(Arc::new(
                super::vector::details::infer_vector_index_details(dataset, output).await?,
            ));
        }
        if let Some(details) = &mut output.index_details {
            *details = configuration_details(details)?;
            if !self.indexed.is_empty() {
                set_overlay_freshness(
                    Arc::make_mut(details),
                    self.indexed,
                    self.validation_version,
                )?;
            }
        } else if !self.indexed.is_empty() {
            return Err(lance_core::Error::index(format!(
                "Index segment {} has indexed overlays but no index details",
                output.uuid
            )));
        }
        Ok(())
    }
}

/// Classify the visible overlays using the versions and actual owners of the data
/// entering a physical merge. The output's dataset version still pins its row-address history.
pub fn merged_overlay_freshness<'a>(
    dataset: &Dataset,
    fields: &[i32],
    output_version: u64,
    sources: impl IntoIterator<Item = (&'a IndexMetadata, Option<&'a RoaringBitmap>)>,
    scanned: &[Fragment],
) -> Result<MergeOverlayFreshness> {
    if dataset.manifest().reader_feature_flags & FLAG_UNSTABLE_DATA_OVERLAY_FILES == 0 {
        return Ok(MergeOverlayFreshness::default());
    }
    let sources = sources
        .into_iter()
        .map(|(source, coverage)| Ok((source, coverage, OverlayFreshness::read(source)?)))
        .collect::<Result<Vec<_>>>()?;
    let needs_local_validation = sources.iter().any(|(source, _, facts)| {
        source.base_id.is_some()
            || facts.validation_version().is_some()
            || facts
                .entries()
                .iter()
                .any(|entry| entry.source_versions.is_empty())
    });
    let scanned: RoaringBitmap = scanned
        .iter()
        .filter(|f| !f.overlays.is_empty())
        .map(|f| f.id as u32)
        .collect();
    let mut entries = Vec::new();
    let mut refresh = RowAddrTreeMap::new();
    for fragment in dataset
        .fragments()
        .iter()
        .filter(|f| !f.overlays.is_empty())
    {
        let id = fragment.id as u32;
        let owners: Vec<_> = sources
            .iter()
            .filter(|(_, coverage, _)| coverage.is_none_or(|covered| covered.contains(id)))
            .collect();
        if owners.is_empty() && !scanned.contains(id) {
            continue;
        }
        let mut refresh_offsets = RoaringBitmap::new();
        let mut overlays = Vec::new();
        for overlay in &fragment.overlays {
            let stale = owners.iter().any(|(source, _, freshness)| {
                freshness.is_stale(id, overlay, source.dataset_version)
            });
            if stale == (overlay.committed_version > output_version) {
                continue;
            }
            if stale {
                refresh_offsets |=
                    stale_overlay_offsets(std::iter::once(overlay), fields, dataset.schema())?;
            } else if overlay
                .data_file
                .fields
                .iter()
                .any(|&field| field >= 0 && field_affects_index(field, fields, dataset.schema()))
            {
                overlays.push(IndexedDataOverlay {
                    committed_version: overlay.committed_version,
                    path: overlay.data_file.path.clone(),
                    base_id: overlay.data_file.base_id,
                });
            }
        }
        if !overlays.is_empty() {
            overlays.sort_unstable_by(|a, b| {
                (a.committed_version, &a.path, a.base_id).cmp(&(
                    b.committed_version,
                    &b.path,
                    b.base_id,
                ))
            });
            overlays.dedup();
            let mut versions = Vec::new();
            if scanned.contains(id) {
                versions.push(dataset.manifest().version);
            }
            for (source, _, freshness) in owners {
                let snapshots = freshness.source_versions(id, &source.dataset_version);
                if source.base_id.is_some() || snapshots.is_empty() {
                    versions.clear();
                    break;
                }
                versions.extend_from_slice(snapshots);
            }
            versions.sort_unstable();
            versions.dedup();
            entries.push(IndexOverlayFreshness {
                fragment_id: id,
                source_versions: versions,
                overlays,
            });
        }
        if !refresh_offsets.is_empty() {
            refresh.insert_bitmap(id, refresh_offsets);
        }
    }
    entries.sort_unstable_by_key(|entry| entry.fragment_id);
    Ok(MergeOverlayFreshness {
        validation_version: (needs_local_validation && !entries.is_empty())
            .then_some(dataset.manifest().version),
        indexed: entries,
        refresh,
    })
}
