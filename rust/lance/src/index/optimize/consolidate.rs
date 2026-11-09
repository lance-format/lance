// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded vector consolidation using the existing optimize execution stages.

use super::*;
use crate::dataset::fragment::FileFragment;
use crate::index::vector::StageParams;
use lance_index::vector::VectorIndex;
use lance_index::vector::bq::builder::RabitQuantizer;

/// Whether `options` ask for a consolidation, bounded or of selected segments.
pub(super) fn is_consolidation(options: &OptimizeOptions) -> bool {
    options.max_rows_to_consolidate.is_some() || options.segments_to_consolidate.is_some()
}

pub(super) fn validate_options(options: &OptimizeOptions) -> Result<()> {
    if let Some(selected) = &options.segments_to_consolidate
        && (selected.is_empty()
            || selected.iter().collect::<HashSet<_>>().len() != selected.len()
            || options
                .index_names
                .as_ref()
                .is_none_or(|names| names.len() != 1)
            || options.max_rows_to_consolidate.is_some())
    {
        return Err(Error::invalid_input(
            "a segment selection needs distinct segments and exactly one index name, and takes no row budget",
        ));
    }
    if !is_consolidation(options) {
        return Ok(());
    }
    // Only a selection packs its output into several segments.
    let segment_limit_allowed = options.segments_to_consolidate.is_some();
    if options.max_rows_to_consolidate == Some(0)
        || options.max_rows_per_task == Some(0)
        || options.max_rows_per_segment == Some(0)
        || options.retrain
        || options.enable_partition_rebalancing
        || options.num_indices_to_merge.is_some()
        || (options.max_rows_per_segment.is_some() && !segment_limit_allowed)
    {
        return Err(Error::invalid_input(
            "consolidation requires positive row limits, cannot be combined with retrain, partition rebalancing or num_indices_to_merge, and takes max_rows_per_segment only with a segment selection",
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
    // A selection is not bounded: the caller chose the work.
    let budget = options.max_rows_to_consolidate.unwrap_or(u64::MAX);
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
    let explicit = options.segments_to_consolidate.as_ref();
    let candidates = match explicit {
        Some(uuids) => {
            let found = segments
                .iter()
                .copied()
                .filter(|segment| uuids.contains(&segment.uuid))
                .collect::<Vec<_>>();
            if found.len() < uuids.len() {
                log::warn!(
                    "Consolidation of index '{}' skips {} selected segments it no longer has",
                    reference.name,
                    uuids.len() - found.len()
                );
            }
            found
        }
        None => segments.to_vec(),
    };
    let mut selected = Vec::new();
    let mut pairs = Vec::new();
    let mut coverage = RoaringBitmap::new();
    let mut rows = 0u64;
    let mut fragment_rows = BTreeMap::new();
    for segment in candidates.iter().rev() {
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
            let live = count.saturating_sub(fragment.count_deletions().await? as u64);
            fragment_rows.insert(fragment.id() as u32, (count, live));
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
    if selected.len() < if explicit.is_some() { 1 } else { 2 } {
        return Ok(());
    }
    selected.reverse();
    pairs.reverse();
    let newest = selected.last().expect("a nonempty selection").clone();
    if selected
        .iter()
        .any(|segment| segment.base_id.is_some() || segment.index_version != newest.index_version)
    {
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
    let newest_index = pairs
        .last()
        .expect("one index per selected segment")
        .1
        .clone();
    let logical = LogicalVectorIndex::try_new(newest.name.clone(), field_path.clone(), pairs)?;
    let shared = vector_segment_compatibility(&logical.as_ivf()?, "planning consolidation")?
        == VectorSegmentCompatibility::SharedModel;
    if shared && explicit.is_none() {
        plan.merges.push(OptimizeIndexMerge {
            plan_id: plan.plan_id,
            read_version: plan.read_version,
            group: plan.merges.len() as u32,
            old_segments: selected,
            replaces: Vec::new(),
        });
        return Ok(());
    }
    let inputs = coverage
        .iter()
        .map(|id| (id, fragment_rows[&id]))
        .collect::<Vec<_>>();
    // A selection gives a fragment larger than a limit a task, or an output, of
    // its own.
    if explicit.is_none()
        && let Some(limit) = options.max_rows_per_task
        && inputs.iter().any(|(_, (physical, _))| *physical > limit)
    {
        return Err(Error::invalid_input(
            "a selected fragment exceeds max_rows_per_task; consolidation cannot split a fragment",
        ));
    }
    // Encoding rows under a model trained on other data loses recall, so a
    // selection whose models differ is retrained on its own rows. A shared
    // model is reused; re-encoding under it still drops deleted rows.
    let reference = if shared || explicit.is_none() {
        newest
    } else {
        let params = selection_params(&newest, newest_index.as_ref())?;
        let fragments = coverage.iter().collect::<Vec<_>>();
        train_model_segment(
            dataset,
            &field_path,
            &newest,
            &params,
            Some(&fragments),
            options.progress.clone(),
        )
        .await?
    };
    let replaces = selected
        .iter()
        .map(|segment| segment.uuid)
        .collect::<Vec<_>>();
    // An output is sized by the live rows it holds, a task by the physical
    // rows it reads.
    let outputs = inputs
        .iter()
        .map(|&(id, (physical, live))| ((id, physical), live))
        .collect();
    let task_limit = [options.max_rows_per_task, options.max_rows_per_segment]
        .into_iter()
        .flatten()
        .min();
    for output in pack_next_fit(outputs, options.max_rows_per_segment) {
        let group = plan.merges.len() as u32;
        for fragments in pack_next_fit(output, task_limit) {
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
        // Every output lists every source: together they replace them.
        plan.merges.push(OptimizeIndexMerge {
            plan_id: plan.plan_id,
            read_version: plan.read_version,
            group,
            old_segments: Vec::new(),
            replaces: replaces.clone(),
        });
    }
    Ok(())
}

/// The build parameters of `newest`, with the partition count left for the
/// builder to size from the rows it trains on. IVF_RQ keeps the rotation the
/// segments of one index share.
fn selection_params(newest: &IndexMetadata, index: &dyn VectorIndex) -> Result<VectorIndexParams> {
    let mut params = fresh_vector_segment_params(newest, index)?;
    for stage in &mut params.stages {
        match stage {
            StageParams::Ivf(ivf) => {
                ivf.num_partitions = None;
                ivf.centroids = None;
            }
            StageParams::RQ(rq) => {
                let quantizer: RabitQuantizer = index.quantizer().try_into()?;
                rq.rotation = Some(quantizer.metadata_ref().clone());
            }
            _ => {}
        }
    }
    Ok(params)
}

/// Reanchor only when the same sources still own the surviving coverage. A
/// concurrent append is disjoint; a delete may reduce coverage without making
/// the staged files unusable. A source replaced by another merge needs replanning.
///
/// The outputs of one index jointly replace its sources: together they must
/// cover every surviving source fragment exactly once.
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
    let incomplete = || {
        Error::invalid_input(
            "consolidation result must cover every surviving source fragment exactly",
        )
    };
    let mut checked = HashSet::new();
    let mut expected: HashMap<String, RoaringBitmap> = HashMap::new();
    for result in results.iter() {
        if result.replaces.is_empty() {
            return Err(incomplete());
        }
        for uuid in &result.replaces {
            if !checked.insert(*uuid) {
                continue;
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
            *expected.entry(result.segment.name.clone()).or_default() |= live;
        }
    }
    let mut actual: HashMap<String, RoaringBitmap> = HashMap::new();
    for result in results.iter_mut() {
        let coverage = result
            .segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .ok_or_else(|| Error::invalid_input("consolidation result has unknown coverage"))?;
        let covered = actual.entry(result.segment.name.clone()).or_default();
        if !covered.is_disjoint(&coverage) {
            return Err(Error::invalid_input(
                "consolidation results replace a source twice",
            ));
        }
        *covered |= &coverage;
        result.segment.fragment_bitmap = Some(coverage);
        result
            .replaces
            .retain(|uuid| current.iter().any(|source| source.uuid == *uuid));
    }
    let none = RoaringBitmap::new();
    if expected
        .keys()
        .chain(actual.keys())
        .any(|name| expected.get(name).unwrap_or(&none) != actual.get(name).unwrap_or(&none))
    {
        return Err(incomplete());
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
