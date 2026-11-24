// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::execution::SendableRecordBatchStream;
use lance_index::frag_reuse::CompactFragReuseIndex;
use lance_index::scalar::CreatedIndex;
use lance_index::scalar::bloomfilter::{
    BloomFilterIndex, merge_bloomfilter_indices, merge_bloomfilter_indices_with_new_data,
    read_bloomfilter_file_info, read_bloomfilter_zone_fragments,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;

use super::segment_merge::{SegmentMerge, downcast_sources, open_dyn_sources};
use crate::index::frag_reuse::StagedRemappingPlans;
use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Merge one caller-defined group of source BloomFilter segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "BloomFilter", &segments)?;
    let fragment_filters = effective_coverage(dataset, merge.segments())?;
    let fragment_bitmap = fragment_filters
        .iter()
        .fold(RoaringBitmap::new(), |acc, effective| acc | effective);

    let (new_uuid, new_store) = merge.new_store(dataset)?;
    let created_index = open_and_merge_segments(
        dataset,
        merge.field_path(),
        merge.segments(),
        None,
        &new_store,
        &fragment_filters,
        staged,
    )
    .await?;

    Ok(merge.build_metadata(
        new_uuid,
        merge.dataset_version(),
        fragment_bitmap,
        created_index,
    ))
}

/// The live fragments of each segment's coverage, the zones a merge keeps.
pub(in crate::index) fn effective_coverage(
    dataset: &Dataset,
    segments: &[&IndexMetadata],
) -> Result<Vec<RoaringBitmap>> {
    let dataset_fragments = dataset.fragment_bitmap.as_ref();
    segments
        .iter()
        .map(|segment| {
            segment
                .effective_fragment_bitmap(dataset_fragments)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "CreateIndex: segment {} is missing fragment coverage",
                        segment.uuid
                    ))
                })
        })
        .collect()
}

/// Merge the zones `segments` hold over `fragment_filters`, and the zones
/// trained from `new_data` when there is any, into `new_store`.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: Option<SendableRecordBatchStream>,
    new_store: &LanceIndexStore,
    fragment_filters: &[RoaringBitmap],
    staged: Option<&StagedRemappingPlans>,
) -> Result<CreatedIndex> {
    let covers_nothing = fragment_filters.iter().all(RoaringBitmap::is_empty);
    let mut selected_segments = Vec::with_capacity(segments.len());
    let mut selected_filters = Vec::with_capacity(segments.len());
    for (position, (segment, effective)) in segments.iter().zip(fragment_filters).enumerate() {
        if effective.is_empty() && !(covers_nothing && position == 0) {
            continue;
        }
        selected_segments.push(*segment);
        selected_filters.push(effective);
    }
    let opened = open_dyn_sources(dataset, field_path, &selected_segments, staged).await?;
    let opened = downcast_sources::<BloomFilterIndex>(&opened, &selected_segments, "BloomFilter")?;
    let source_indices = opened
        .iter()
        .zip(&selected_filters)
        .map(|(source, effective)| (*source, *effective))
        .collect::<Vec<_>>();

    match new_data {
        Some(new_data) => {
            merge_bloomfilter_indices_with_new_data(&source_indices, new_data, new_store).await
        }
        None => merge_bloomfilter_indices(&source_indices, new_store).await,
    }
}

/// Whether the zones of `segments` merge with zones built with the parameters
/// of `reference`: each segment records its null rows, which a merge would
/// otherwise drop for every row it holds, and was built with those parameters.
pub(in crate::index) async fn segments_merge_by_zones(
    dataset: &Dataset,
    reference: &IndexMetadata,
    segments: &[&IndexMetadata],
) -> Result<bool> {
    let file_info = async |segment: &IndexMetadata| {
        let store = LanceIndexStore::from_dataset_for_existing(dataset, segment).await?;
        read_bloomfilter_file_info(&store).await
    };
    let params = file_info(reference).await?.params;
    for segment in segments {
        let info = file_info(segment).await?;
        if !info.has_null_rows || info.params != params {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether any of `segments` holds zones of a fragment outside its coverage
/// that still answer queries: an in-place column rewrite takes the fragment
/// out of the segment and leaves its zones and null rows in the file. Such
/// zones answer while the fragment is live, and after a compaction moves its
/// rows too, since `frag_reuse_index` translates them onto the new fragment.
pub(in crate::index) async fn segments_hold_rewritten_fragments(
    dataset: &Dataset,
    frag_reuse_index: Option<&CompactFragReuseIndex>,
    segments: &[&IndexMetadata],
) -> Result<bool> {
    let mut answering = dataset.fragment_bitmap.as_ref().clone();
    if let Some(frag_reuse_index) = frag_reuse_index {
        answering.extend(
            frag_reuse_index
                .details
                .versions
                .iter()
                .flat_map(|version| version.old_frag_ids())
                .map(|fragment_id| fragment_id as u32),
        );
    }
    for segment in segments {
        let store = LanceIndexStore::from_dataset_for_existing(dataset, segment).await?;
        let mut outside = read_bloomfilter_zone_fragments(&store).await?;
        outside &= &answering;
        if let Some(covered) = &segment.fragment_bitmap {
            outside -= covered;
        }
        if !outside.is_empty() {
            return Ok(true);
        }
    }
    Ok(false)
}
