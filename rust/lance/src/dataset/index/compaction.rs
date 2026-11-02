// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Eligibility and rewrite dependencies for vector segment compaction.

use crate::Dataset;
use crate::index::vector::ivf::compaction_vector_segment_merge_groups;
use crate::index::{DatasetIndexInternalExt, LogicalVectorIndex};
use futures::{StreamExt, stream};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pb::{VectorIndexDetails, vector_index_details::Compression};
use lance_table::format::IndexMetadata;
use lance_table::transaction::RewriteGroup;
use roaring::RoaringBitmap;
use std::collections::HashMap;

pub async fn vector_segment_merge_groups_for_compaction<'a>(
    dataset: &Dataset,
    segments: &[&'a IndexMetadata],
) -> Vec<Vec<&'a IndexMetadata>> {
    let singleton_groups = || segments.iter().map(|segment| vec![*segment]).collect();
    if segments.len() < 2 {
        return singleton_groups();
    }
    let Some(reference) = segments.iter().find(|segment| is_merge_candidate(segment)) else {
        return singleton_groups();
    };

    let Ok(column) = dataset.schema().field_path(reference.fields[0]) else {
        return singleton_groups();
    };
    // Manifest details reject unsupported layouts before opening any index files.
    // Missing ownership is also insufficient for a many-to-one replacement.
    let eligible = segments
        .iter()
        .copied()
        .filter(|segment| is_merge_candidate(segment) && segment.fields == reference.fields)
        .collect::<Vec<_>>();
    if eligible.len() < 2 {
        return singleton_groups();
    }
    let eligible = eligible.into_iter().cloned().collect::<Vec<_>>();
    let opened = stream::iter(eligible)
        .map(|segment| {
            let column = &column;
            async move {
                dataset
                    .open_vector_index(column, &segment.uuid, &NoOpMetricsCollector)
                    .await
                    .map(|index| (segment, index))
            }
        })
        .buffered(dataset.object_store.io_parallelism())
        .collect::<Vec<_>>()
        .await;
    let opened = opened
        .into_iter()
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    if opened.len() < 2 {
        return singleton_groups();
    }
    let Ok(logical_index) = LogicalVectorIndex::try_new(reference.name.clone(), column, opened)
    else {
        return singleton_groups();
    };
    let Ok(ivf) = logical_index.as_ivf() else {
        return singleton_groups();
    };

    let mut segments_by_id = segments
        .iter()
        .map(|segment| (segment.uuid, *segment))
        .collect::<HashMap<_, _>>();
    let mut groups = compaction_vector_segment_merge_groups(&ivf)
        .into_iter()
        .filter_map(|group| {
            let group = group
                .into_iter()
                .filter_map(|segment_id| segments_by_id.remove(&segment_id))
                .collect::<Vec<_>>();
            (!group.is_empty()).then_some(group)
        })
        .collect::<Vec<_>>();

    // Rejected metadata still contributes its physical coverage boundary.
    groups.extend(segments_by_id.into_values().map(|segment| vec![segment]));
    groups
}

/// Connect source segments only when a completed rewrite consumes fragments
/// from both. Bitmap intersections keep this independent of dataset row count.
pub(super) fn rewrite_connected_segments<'a>(
    segments: &[&'a IndexMetadata],
    rewrite_groups: &[RewriteGroup],
) -> Vec<Vec<&'a IndexMetadata>> {
    let mut components = segments
        .iter()
        .map(|segment| vec![*segment])
        .collect::<Vec<_>>();
    for group in rewrite_groups {
        let fragments = group
            .old_fragments
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect::<RoaringBitmap>();
        let mut connected = Vec::new();
        let mut separate = Vec::with_capacity(components.len());
        for component in components {
            if component.iter().any(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|coverage| !coverage.is_disjoint(&fragments))
            }) {
                connected.extend(component);
            } else {
                separate.push(component);
            }
        }
        if !connected.is_empty() {
            separate.push(connected);
        }
        components = separate;
    }
    components
}

/// Manifest settings can reject a merge, but stored models must still agree.
pub fn is_merge_candidate(segment: &IndexMetadata) -> bool {
    segment.fields.len() == 1
        && segment.fragment_bitmap.is_some()
        && segment
            .index_details
            .as_ref()
            .filter(|details| details.type_url.ends_with("VectorIndexDetails"))
            .and_then(|details| details.to_msg::<VectorIndexDetails>().ok())
            .is_some_and(|details| {
                details.hnsw_index_config.is_none()
                    && matches!(
                        details.compression,
                        Some(Compression::Flat(_) | Compression::Sq(_))
                    )
            })
}

/// Split possible windows at coverage boundaries known without model inspection.
pub fn constrain_candidate_windows(
    windows: &[RoaringBitmap],
    boundaries: &[RoaringBitmap],
) -> Vec<RoaringBitmap> {
    let mut constrained = Vec::new();
    for window in windows {
        let mut current = RoaringBitmap::new();
        let mut previous = Vec::new();
        for fragment in window {
            let signature = boundaries
                .iter()
                .map(|coverage| coverage.contains(fragment))
                .collect::<Vec<_>>();
            if !current.is_empty() && previous != signature {
                if current.len() > 1 {
                    constrained.push(current);
                }
                current = RoaringBitmap::new();
            }
            current.insert(fragment);
            previous = signature;
        }
        if current.len() > 1 {
            constrained.push(current);
        }
    }
    constrained
}
