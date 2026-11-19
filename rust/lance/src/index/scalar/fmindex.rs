// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use uuid::Uuid;

use lance_table::format::IndexMetadata;

use super::segment_merge::SegmentMerge;
use crate::{Dataset, Result};

/// Merge one caller-defined group of source FM-Index segments into a single segment.
///
/// FM-Index merge requires rebuilding from source text — there is no cheap way
/// to combine two BWT structures. This function re-reads text data from the
/// dataset for all fragments covered by the source segments and builds a fresh
/// FM-Index over the combined data.
///
/// As an exception, a single source segment whose fragment coverage is still
/// fully live is returned as-is: rebuilding it would produce equivalent content
/// over the same fragments, which makes distributed builds (one uncommitted
/// segment per worker, merged 1:1 into final segments) pay for every segment
/// twice. `is_rebuild_required` disables that passthrough when the caller has
/// moved coverage into the current fragment space and the stored row addresses
/// must be rebuilt to match it.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    is_rebuild_required: bool,
    refresh: &super::merge::MergeRefresh,
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "FM-Index", &segments)?;
    let column = merge.field_path();
    let mut fragment_bitmap = merge.stored_coverage_union()? | refresh.fragments();
    fragment_bitmap &= dataset.table_state.fragment_bitmap.as_ref();

    if fragment_bitmap.is_empty() {
        // All covered fragments have been retired; produce an empty index.
        let new_uuid = Uuid::new_v4();
        let created_index = super::build_scalar_index(
            dataset,
            column,
            new_uuid,
            &lance_index::scalar::ScalarIndexParams::for_builtin(
                lance_index::scalar::BuiltinIndexType::Fm,
            ),
            false,
            None,
            None,
            Arc::new(lance_index::progress::NoopIndexBuildProgress),
        )
        .await?;

        return Ok(merge.build_metadata(new_uuid, fragment_bitmap, created_index));
    }

    if !is_rebuild_required
        && refresh.is_empty()
        && segments.len() == 1
        && segments[0].fragment_bitmap.as_ref() == Some(&fragment_bitmap)
    {
        return Ok(segments[0].clone());
    }

    let fragment_ids: Vec<u32> = fragment_bitmap.iter().collect();
    let new_uuid = Uuid::new_v4();

    let created_index = super::build_scalar_index(
        dataset,
        column,
        new_uuid,
        &lance_index::scalar::ScalarIndexParams::for_builtin(
            lance_index::scalar::BuiltinIndexType::Fm,
        ),
        true,
        Some(fragment_ids),
        None,
        Arc::new(lance_index::progress::NoopIndexBuildProgress),
    )
    .await?;

    Ok(merge.build_metadata(new_uuid, fragment_bitmap, created_index))
}
