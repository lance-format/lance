// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use datafusion::physical_plan::SendableRecordBatchStream;
use lance_index::frag_reuse::CompactFragReuseIndexHandle;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::ngram::NGramIndex;
use lance_index::scalar::{
    BuiltinIndexType, CreatedIndex, IndexStore, OldIndexDataFilter, RowIdRemapper,
    ScalarIndexParams,
};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use super::segment_merge::SegmentMerge;
use crate::{
    Dataset, Error, Result, dataset::index::LanceIndexStoreExt,
    index::loading::DatasetIndexInternalExt,
};

async fn collect_ngram_segment_stores(
    dataset: &Dataset,
    segments: &[IndexMetadata],
) -> Result<Vec<Arc<dyn IndexStore>>> {
    let mut stores: Vec<Arc<dyn IndexStore>> = Vec::with_capacity(segments.len());
    for segment in segments {
        let store = LanceIndexStore::from_dataset_for_existing(dataset, segment).await?;
        stores.push(Arc::new(store));
    }
    Ok(stores)
}

/// Merge one caller-defined group of source NGram segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "NGram", &segments)?;
    let field_path = merge.field_path();

    let frag_reuse_index = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
    let has_retired_coverage = merge.segments().iter().any(|segment| {
        segment
            .deleted_fragment_bitmap(&dataset.fragment_bitmap)
            .is_some_and(|retired| !retired.is_empty())
    });
    let requires_rebuild = frag_reuse_index.as_ref().is_some_and(|frag_reuse_index| {
        crate::index::append::fragment_reuse_affects_segments(
            frag_reuse_index,
            merge.segments().iter().copied(),
        )
    });
    if has_retired_coverage && !requires_rebuild {
        return Err(Error::invalid_input(
            "NGram merge_segments: source segments cover retired fragments but no applicable \
             fragment-reuse mapping is available; rebuild the affected segments from the current \
             dataset"
                .to_string(),
        ));
    }
    let (new_uuid, created_index, dataset_version, fragment_bitmap) = if requires_rebuild {
        let mut fragment_bitmap = merge.stored_coverage_union()?;
        frag_reuse_index
            .as_ref()
            .expect("requires_rebuild implies a fragment-reuse index")
            .remap_fragment_bitmap(&mut fragment_bitmap)?;
        fragment_bitmap &= dataset.fragment_bitmap.as_ref();
        let new_uuid = Uuid::new_v4();
        (
            new_uuid,
            super::build_scalar_index(
                dataset,
                field_path,
                new_uuid,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::NGram),
                true,
                Some(fragment_bitmap.iter().collect()),
                None,
                Arc::new(NoopIndexBuildProgress),
            )
            .await?,
            dataset.manifest.version,
            fragment_bitmap,
        )
    } else {
        // `open_and_merge_segments` refuses a row-id-domain segment on a
        // stable-row-id dataset, and without stable row ids the two domains
        // coincide, so the filters can always be address-domain.
        let (fragment_bitmap, old_data_filters) =
            merge.per_segment_filters(dataset, staged, true).await?;
        let (new_uuid, new_store) = merge.new_store(dataset)?;
        (
            new_uuid,
            open_and_merge_segments(
                dataset,
                merge.segments(),
                None,
                &new_store,
                &old_data_filters,
                staged,
            )
            .await?,
            merge.dataset_version(),
            fragment_bitmap,
        )
    };

    Ok(merge.build_metadata(new_uuid, dataset_version, fragment_bitmap, created_index))
}

/// Merge the given NGram segments with optional newly appended data into a single
/// canonical segment. Pass `None` for `new_data` for a pure consolidation.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    segments: &[&IndexMetadata],
    new_data: Option<SendableRecordBatchStream>,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<CreatedIndex> {
    // The NGram merge reads its sources' spill files directly and translates
    // them through the synchronous v0 remapper only; a tagged history has no
    // such remapper, so a segment whose rows need translation would be merged
    // with stale addresses against filters expressed in the translated
    // domain. Refuse instead of silently emptying the merge; the segment
    // keeps translating at query time and a rebuild catches it up.
    if crate::index::append::tagged_segment_coverage(dataset, segments, staged)
        .await?
        .is_some()
        && let Some(segment) = segments.iter().find(|segment| {
            use crate::index::frag_reuse::SegmentRemappingPlan;
            match staged.and_then(|plans| plans.get(&segment.uuid)) {
                // A staged segment's bitmap already holds its planned live
                // coverage; the plan says whether its rows translate.
                Some(SegmentRemappingPlan::Translate { .. }) => true,
                Some(_) => false,
                None => segment
                    .fragment_bitmap
                    .as_ref()
                    .is_none_or(|bitmap| !bitmap.is_subset(dataset.fragment_bitmap.as_ref())),
            }
        })
    {
        return Err(Error::not_supported(format!(
            "NGram segment {} needs address translation under the tagged fragment reuse \
             history and cannot be merged; rebuild the index instead",
            segment.uuid
        )));
    }
    let results_are_row_addresses = super::shared_segment_domain(dataset, "NGram", segments)?;
    let segments = segments.iter().map(|&s| s.clone()).collect::<Vec<_>>();
    let segment_stores = collect_ngram_segment_stores(dataset, &segments).await?;
    let frag_reuse_index = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
    let frag_reuse_index = frag_reuse_index
        .map(|index| Arc::new(CompactFragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
    NGramIndex::merge_segments_with_remapper(
        &segment_stores,
        new_data,
        new_store,
        old_data_filters,
        frag_reuse_index,
        results_are_row_addresses,
    )
    .await
}
