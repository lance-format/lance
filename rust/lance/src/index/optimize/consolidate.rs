// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded vector consolidation using the existing optimize execution stages.

use super::*;
use crate::dataset::fragment::FileFragment;
use lance_index::vector::VectorIndex;

pub(super) fn validate_options(options: &OptimizeOptions) -> Result<()> {
    if let Some(selected) = &options.consolidate_segments {
        let one_index = options
            .index_names
            .as_ref()
            .is_some_and(|names| names.len() == 1);
        if selected.is_empty()
            || !one_index
            || options.max_rows_to_consolidate.is_none()
            || options.max_rows_per_segment == Some(0)
        {
            return Err(Error::invalid_input(
                "consolidating selected segments requires at least one segment, exactly one index name, a row budget, and a positive max_rows_per_segment when set",
            ));
        }
    }
    let Some(budget) = options.max_rows_to_consolidate else {
        return Ok(());
    };
    if budget == 0
        || options.max_rows_per_task == Some(0)
        || options.retrain
        || options.enable_partition_rebalancing
        || options.num_indices_to_merge.is_some()
        || (options.max_rows_per_segment.is_some() && options.consolidate_segments.is_none())
    {
        return Err(Error::invalid_input(
            "consolidation requires a positive row budget and task limit, and cannot be combined with retrain, partition rebalancing, num_indices_to_merge, or, without selected segments, max_rows_per_segment",
        ));
    }
    Ok(())
}

pub(super) async fn plan_index(
    plan: &mut OptimizeIndicesPlan,
    dataset: &Dataset,
    segments: &[&IndexMetadata],
    options: &OptimizeOptions,
    tagged: bool,
) -> Result<()> {
    if options.consolidate_segments.is_some() {
        return plan_selected(plan, dataset, segments, options, tagged).await;
    }
    let budget = options
        .max_rows_to_consolidate
        .expect("consolidation requested");
    let named = options.index_names.is_some();
    let reference = segments.last().expect("nonempty index");
    if !segment_has_vector_details(reference) || tagged {
        if named {
            return Err(Error::not_supported(
                "consolidation requires a vector index without tagged fragment reuse",
            ));
        }
        return Ok(());
    }
    let (field_path, _) = index_field_path(dataset, reference).await?;
    let mut selected = Vec::new();
    let mut pairs = Vec::new();
    let mut coverage = RoaringBitmap::new();
    let mut rows = 0u64;
    let mut fragment_rows = BTreeMap::new();
    for segment in segments.iter().rev() {
        let effective = segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .ok_or_else(|| {
                Error::not_supported("consolidation requires known fragment coverage")
            })?;
        if effective.is_empty() {
            continue;
        }
        let mut segment_rows = 0u64;
        for fragment in dataset.get_fragments_from_ids(&effective.iter().collect::<Vec<_>>())? {
            let count = fragment.physical_rows().await? as u64;
            segment_rows = segment_rows
                .checked_add(count)
                .ok_or_else(|| Error::invalid_input("consolidation physical row count overflow"))?;
            fragment_rows.insert(fragment.id() as u32, count);
            if segment_rows > budget - rows {
                break;
            }
        }
        if segment_rows > budget - rows {
            break;
        }
        let index = dataset
            .open_vector_index_from_metadata_with_plan(
                &field_path,
                segment,
                Some(&SegmentRemappingPlan::Identity),
                OpenPurpose::Maintenance,
                &NoOpMetricsCollector,
            )
            .await?;
        // Direct merges also read retired postings, whereas re-encoding reads
        // the raw fragments (including rows with null/unindexable vectors).
        let segment_rows = segment_rows.max(index.num_rows());
        if segment_rows > budget - rows {
            break;
        }
        if index.as_any().is::<LegacyIvfIndex>() {
            return Err(Error::not_supported(
                "consolidation requires the current vector index format",
            ));
        }
        if !coverage.is_disjoint(&effective) {
            return Err(Error::not_supported(
                "consolidation requires disjoint source coverage",
            ));
        }
        rows += segment_rows;
        coverage |= effective;
        pairs.push(((*segment).clone(), index));
        selected.push((*segment).clone());
    }
    if selected.len() < 2 {
        return Ok(());
    }
    selected.reverse();
    pairs.reverse();
    let reference = selected.last().expect("at least two selected segments");
    if selected.iter().any(|segment| {
        segment.base_id.is_some() || segment.index_version != reference.index_version
    }) {
        return Err(Error::not_supported(
            "consolidation requires local segments with matching on-disk versions",
        ));
    }
    if let Some(reuse) = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?
        && fragment_reuse_affects_segments(&reuse, selected.iter())
    {
        return Err(Error::not_supported(
            "consolidation requires pending compaction remaps to be applied first",
        ));
    }
    let logical = LogicalVectorIndex::try_new(reference.name.clone(), field_path, pairs)?;
    let shared = vector_segment_compatibility(&logical.as_ivf()?, "planning consolidation")?
        == VectorSegmentCompatibility::SharedModel;
    let group = plan.merges.len() as u32;
    if shared {
        plan.merges.push(OptimizeIndexMerge {
            plan_id: plan.plan_id,
            read_version: plan.read_version,
            group,
            old_segments: selected,
            replaces: Vec::new(),
        });
        return Ok(());
    }
    let inputs = coverage
        .iter()
        .map(|id| (id, fragment_rows[&id]))
        .collect::<Vec<_>>();
    if let Some(limit) = options.max_rows_per_task
        && inputs.iter().any(|(_, rows)| *rows > limit)
    {
        return Err(Error::invalid_input(
            "a selected fragment exceeds max_rows_per_task; consolidation cannot split a fragment",
        ));
    }
    for fragments in pack_next_fit(inputs, options.max_rows_per_task) {
        plan.tasks.push(OptimizeIndexTask {
            plan_id: plan.plan_id,
            read_version: plan.read_version,
            group,
            fragments,
            reference: reference.clone(),
            old_segments: Vec::new(),
            num_indices_to_merge: Some(0),
            enable_partition_rebalancing: false,
            retrain: false,
        });
    }
    plan.merges.push(OptimizeIndexMerge {
        plan_id: plan.plan_id,
        read_version: plan.read_version,
        group,
        old_segments: Vec::new(),
        replaces: selected.iter().map(|segment| segment.uuid).collect(),
    });
    Ok(())
}

/// One selected segment: what it covers now, and the rows it holds.
struct SelectedSource {
    segment: IndexMetadata,
    /// `(fragment id, physical rows)` of every fragment it still covers.
    fragments: Vec<(u32, u64)>,
    /// Rows of those fragments that are not deleted: what re-encoding writes.
    live_rows: u64,
}

/// Plan [`OptimizeOptions::consolidate_segments`]: re-encode exactly the named
/// segments, wherever they sit in the index, packed whole into outputs of at
/// most `max_rows_per_segment` live rows. Each output replaces only the
/// segments packed into it, so commit validation checks it like any other
/// consolidation result.
async fn plan_selected(
    plan: &mut OptimizeIndicesPlan,
    dataset: &Dataset,
    segments: &[&IndexMetadata],
    options: &OptimizeOptions,
    tagged: bool,
) -> Result<()> {
    let wanted = options
        .consolidate_segments
        .as_ref()
        .expect("selected consolidation requested");
    let budget = options
        .max_rows_to_consolidate
        .expect("validated with selected segments");
    let mut unique = HashSet::with_capacity(wanted.len());
    if let Some(repeated) = wanted.iter().find(|uuid| !unique.insert(**uuid)) {
        return Err(Error::invalid_input(format!(
            "consolidation selects segment {repeated} more than once"
        )));
    }
    // Index order, which is the order the outputs take. A selected segment the
    // index no longer has was removed since the selection was made -- a delete
    // that empties a segment's fragments drops the segment -- so there is
    // nothing left of it to rewrite.
    let selected = segments
        .iter()
        .copied()
        .filter(|segment| unique.contains(&segment.uuid))
        .collect::<Vec<_>>();
    if selected.len() < unique.len() {
        log::warn!(
            "Consolidation of index '{}' skips {} selected segments the index no longer has",
            segments[0].name,
            unique.len() - selected.len()
        );
    }
    if selected.is_empty() {
        return Ok(());
    }
    // The model appends use: the newest segment that still covers rows. The
    // outputs commit at the end of the index, so re-encoding with it keeps that
    // model the newest even when the selection leaves its segment out.
    let (live, _) = partition_dormant_vector_segments(dataset, segments, None);
    let Some(reference) = live.last().map(|segment| (*segment).clone()) else {
        return Ok(());
    };
    if !segment_has_vector_details(&reference) || tagged {
        return Err(Error::not_supported(
            "consolidation requires a vector index without tagged fragment reuse",
        ));
    }
    let (field_path, _) = index_field_path(dataset, &reference).await?;
    let mut sources = Vec::new();
    let mut pairs = Vec::new();
    let mut coverage = RoaringBitmap::new();
    let mut read = 0u64;
    for segment in selected {
        let effective = segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .ok_or_else(|| {
                Error::not_supported("consolidation requires known fragment coverage")
            })?;
        // Nothing left to rewrite; the newest-suffix selection skips these too.
        if effective.is_empty() {
            log::warn!(
                "Consolidation skips segment {} of index '{}': it covers no fragment left",
                segment.uuid,
                segment.name
            );
            continue;
        }
        if !coverage.is_disjoint(&effective) {
            return Err(Error::not_supported(
                "consolidation requires disjoint source coverage",
            ));
        }
        let mut physical = 0u64;
        let mut live_rows = 0u64;
        let mut fragments = Vec::new();
        for fragment in dataset.get_fragments_from_ids(&effective.iter().collect::<Vec<_>>())? {
            let rows = fragment.physical_rows().await? as u64;
            physical = physical
                .checked_add(rows)
                .ok_or_else(|| Error::invalid_input("consolidation physical row count overflow"))?;
            live_rows = live_rows
                .checked_add(fragment.count_rows(None).await? as u64)
                .ok_or_else(|| Error::invalid_input("consolidation live row count overflow"))?;
            fragments.push((fragment.id() as u32, rows));
        }
        let index = open_current_format(dataset, &field_path, segment).await?;
        // Charged as the newest-segments consolidation charges a segment: the
        // greater of its entries and the rows its fragments hold.
        read = read
            .checked_add(physical.max(index.num_rows()))
            .ok_or_else(|| Error::invalid_input("consolidation physical row count overflow"))?;
        coverage |= effective;
        pairs.push((segment.clone(), index));
        sources.push(SelectedSource {
            segment: segment.clone(),
            fragments,
            live_rows,
        });
    }
    if read > budget {
        return Err(Error::invalid_input(format!(
            "the selected segments hold {read} physical rows, more than the consolidation budget of {budget}"
        )));
    }
    if sources.is_empty() {
        return Ok(());
    }
    if !pairs
        .iter()
        .any(|(segment, _)| segment.uuid == reference.uuid)
    {
        let index = open_current_format(dataset, &field_path, &reference).await?;
        pairs.push((reference.clone(), index));
    }
    if pairs.iter().any(|(segment, _)| {
        segment.base_id.is_some() || segment.index_version != reference.index_version
    }) {
        return Err(Error::not_supported(
            "consolidation requires local segments with matching on-disk versions",
        ));
    }
    if let Some(reuse) = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?
        && fragment_reuse_affects_segments(&reuse, sources.iter().map(|source| &source.segment))
    {
        return Err(Error::not_supported(
            "consolidation requires pending compaction remaps to be applied first",
        ));
    }
    // Every selected segment must answer queries the same way the output will.
    LogicalVectorIndex::try_new(reference.name.clone(), field_path, pairs)?;
    if let Some(limit) = options.max_rows_per_task
        && sources
            .iter()
            .flat_map(|source| &source.fragments)
            .any(|(_, rows)| *rows > limit)
    {
        return Err(Error::invalid_input(
            "a selected fragment exceeds max_rows_per_task; consolidation cannot split a fragment",
        ));
    }
    let outputs = pack_next_fit(
        sources
            .into_iter()
            .map(|source| {
                let rows = source.live_rows;
                (source, rows)
            })
            .collect(),
        options.max_rows_per_segment,
    );
    for output in outputs {
        let group = plan.merges.len() as u32;
        let fragments = output
            .iter()
            .flat_map(|source| source.fragments.iter().copied())
            .collect::<Vec<_>>();
        for fragments in pack_next_fit(fragments, options.max_rows_per_task) {
            plan.tasks.push(OptimizeIndexTask {
                plan_id: plan.plan_id,
                read_version: plan.read_version,
                group,
                fragments,
                reference: reference.clone(),
                old_segments: Vec::new(),
                num_indices_to_merge: Some(0),
                enable_partition_rebalancing: false,
                retrain: false,
            });
        }
        plan.merges.push(OptimizeIndexMerge {
            plan_id: plan.plan_id,
            read_version: plan.read_version,
            group,
            old_segments: Vec::new(),
            replaces: output.iter().map(|source| source.segment.uuid).collect(),
        });
    }
    Ok(())
}

async fn open_current_format(
    dataset: &Dataset,
    field_path: &str,
    segment: &IndexMetadata,
) -> Result<Arc<dyn VectorIndex>> {
    let index = dataset
        .open_vector_index_from_metadata_with_plan(
            field_path,
            segment,
            Some(&SegmentRemappingPlan::Identity),
            OpenPurpose::Maintenance,
            &NoOpMetricsCollector,
        )
        .await?;
    if index.as_any().is::<LegacyIvfIndex>() {
        return Err(Error::not_supported(
            "consolidation requires the current vector index format",
        ));
    }
    Ok(index)
}

/// Reanchor only when the same sources still own the surviving coverage. A
/// concurrent append is disjoint; a delete may reduce coverage without making
/// the staged files unusable. A source replaced by another merge needs replanning.
pub(super) async fn validate_commit(
    dataset: &Dataset,
    planned: &Dataset,
    results: &mut Vec<IndexSegmentResult>,
) -> Result<()> {
    let original = load_all_indices(planned).await?;
    let current = load_all_indices(dataset).await?;
    if current
        .iter()
        .any(lance_table::system_index::frag_reuse::metadata::is_tagged)
    {
        return Err(Error::not_supported(
            "consolidation cannot commit across tagged fragment reuse; replan",
        ));
    }
    let reuse = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
    let mut all_replaced = HashSet::new();
    for result in results.iter_mut() {
        let mut expected = RoaringBitmap::new();
        for uuid in &result.replaces {
            if !all_replaced.insert(*uuid) {
                return Err(Error::invalid_input(
                    "consolidation results replace a source twice",
                ));
            }
            let source = original
                .iter()
                .find(|source| source.uuid == *uuid && source.name == result.segment.name)
                .ok_or_else(|| {
                    Error::invalid_input("consolidation result references an unknown source")
                })?;
            if reuse
                .as_ref()
                .is_some_and(|reuse| fragment_reuse_affects_segments(reuse, [source]))
            {
                return Err(Error::not_supported(
                    "consolidation source has a pending compaction remap; replan",
                ));
            }
            let live = source
                .effective_fragment_bitmap(&dataset.fragment_bitmap)
                .ok_or_else(|| {
                    Error::not_supported("consolidation requires known source coverage")
                })?;
            let Some(now) = current.iter().find(|now| now.uuid == *uuid) else {
                // Append maintenance can remove a fully deleted source. It no
                // longer owns any live rows and needs no replacement at commit.
                if live.is_empty() {
                    continue;
                }
                return Err(Error::invalid_input(format!(
                    "consolidation source {uuid} was replaced; replan before committing"
                )));
            };
            let mut expected_metadata = source.clone();
            expected_metadata.fragment_bitmap = now.fragment_bitmap.clone();
            if &expected_metadata != now
                || now.effective_fragment_bitmap(&dataset.fragment_bitmap) != Some(live.clone())
            {
                return Err(Error::invalid_input(format!(
                    "consolidation source {uuid} changed; replan before committing"
                )));
            }
            // Reanchoring must not skip conflicts from updates/overlays that
            // keep fragment IDs but change the vectors encoded by this output.
            for id in &live {
                let before = planned
                    .manifest
                    .fragments
                    .iter()
                    .find(|fragment| fragment.id == id as u64)
                    .ok_or_else(|| {
                        Error::invalid_input(
                            "consolidation source fragment was not present when planned",
                        )
                    })?;
                let after = dataset
                    .manifest
                    .fragments
                    .iter()
                    .find(|fragment| fragment.id == id as u64)
                    .expect("live fragment is in the manifest");
                let mut unchanged = before.clone();
                unchanged.deletion_file = after.deletion_file.clone();
                if &unchanged != after {
                    return Err(Error::invalid_input(format!(
                        "consolidation source fragment {id} changed; replan before committing"
                    )));
                }
                // Re-encoding omits deleted rows. A restore may make them live
                // again without changing the source UUID or its data files.
                if before.deletion_file.is_some() && before.deletion_file != after.deletion_file {
                    let deleted_before =
                        FileFragment::new(Arc::new(planned.clone()), before.clone())
                            .get_deletion_vector()
                            .await?;
                    let deleted_after = FileFragment::new(Arc::new(dataset.clone()), after.clone())
                        .get_deletion_vector()
                        .await?;
                    if deleted_before.is_some_and(|before| {
                        before.iter().any(|offset| {
                            !deleted_after
                                .as_ref()
                                .is_some_and(|after| after.contains(offset))
                        })
                    }) {
                        return Err(Error::invalid_input(format!(
                            "consolidation source fragment {id} has restored rows; replan before committing"
                        )));
                    }
                }
            }
            expected |= live;
        }
        let actual = result
            .segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .ok_or_else(|| Error::invalid_input("consolidation result has unknown coverage"))?;
        if result.replaces.is_empty() || actual != expected {
            return Err(Error::invalid_input(
                "consolidation result must cover every surviving source fragment exactly",
            ));
        }
        result.segment.fragment_bitmap = Some(actual);
        result
            .replaces
            .retain(|uuid| current.iter().any(|source| source.uuid == *uuid));
    }
    // Fully retired outputs have no work left to publish.
    results.retain(|result| {
        result
            .segment
            .fragment_bitmap
            .as_ref()
            .is_some_and(|coverage| !coverage.is_empty())
    });
    Ok(())
}

#[cfg(test)]
mod tests;
