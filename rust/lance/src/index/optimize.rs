// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index optimization as plan, execute, merge and commit.
//!
//! [`DatasetIndexExt::optimize_indices`] runs these four stages in one
//! process. A distributed engine runs the same stages across workers, in the
//! shape of compaction's plan / execute / commit:
//!
//! ```text
//! driver   plan_optimize_indices(dataset, options) -> OptimizeIndicesPlan { tasks, merges }
//! workers  task.execute(dataset)                   -> Option<IndexSegmentResult>   (one per task)
//! workers  merge.execute(dataset, &task_results)   -> Option<IndexSegmentResult>   (one per merge)
//! driver   commit_optimize_indices(dataset, merged_results, options)
//! ```
//!
//! Without `max_rows_per_task` and `max_rows_per_segment` every index is one
//! task: the single-process optimize over all of its segments and all of its
//! unindexed fragments, with the caller's options, and a merge that passes the
//! result through. That is `optimize_indices` itself, which is why the two
//! paths cannot drift apart.
//!
//! With a bound set, an index whose kind can merge segments (vector, BTree,
//! Bitmap, NGram, Inverted) is split: tasks build one uncommitted segment each
//! over a group of fragments, encoded against the model of a reference
//! segment; merges combine the task results that carry their `group` with the
//! committed segments they list into one segment. A task that is the only
//! input of its merge folds those old segments in itself, in one pass. For a
//! vector index `num_indices_to_merge` unset means append only: the
//! single-process optimize appends a delta too, unless the partitions need
//! rebalancing, which a split build does not do (set `retrain` for that). The
//! other kinds are rebuilt whole by the single-process optimize, so they stay
//! one task.
//!
//! Planning is not always cheap: a split vector index that has to be rebuilt
//! (`retrain`, every segment's fragments gone, or nothing but a definition so
//! far) is trained by the planner, which writes the model as a segment outside
//! the manifest for the tasks to encode against; cleanup reclaims it like any
//! unreferenced index directory.
//!
//! The commit publishes every merged segment in one `CreateIndex` transaction
//! anchored at the plan's `read_version`, so conflict detection covers
//! everything committed since the plan was made, and removes the segments the
//! results replace. Results may be partial: whatever they do not cover stays
//! unindexed for the next optimize.
//!
//! An engine like Spark drives it as two map stages:
//!
//! ```text
//! val plan   = planOptimizeIndices(ds, opts)
//! val built  = sc.parallelize(plan.tasks).flatMap(_.execute(ds)).collect()
//! val merged = sc.parallelize(plan.merges).flatMap(_.execute(ds, built)).collect()
//! commitOptimizeIndices(ds, merged, opts)
//! ```
//!
//! A split build skips, with a warning, what its merge cannot handle: a table
//! with a tagged fragment reuse history, a legacy (v1) vector index and a
//! scalar index that must be rebuilt from the whole column. Naming such an
//! index in `OptimizeOptions::index_names` refuses instead.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use chrono::Utc;
use futures::FutureExt;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::{IndexBuildProgress, noop_progress};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::append::{
    append_vector_segment, expects_definition_only, fragment_reuse_affects_segments,
    index_field_path, is_definition_only_segment, merge_indices_with_unindexed_frags,
    partition_dormant_vector_segments, select_segments_to_merge, tagged_segment_coverage,
};
use super::frag_reuse::{OpenPurpose, SegmentRemappingPlan};
use super::vector::details::vector_params_from_details;
use super::vector::ivf::{
    IVFIndex as LegacyIvfIndex, VectorSegmentCompatibility, optimize_vector_indices,
    select_steady_state_rebalance, vector_segment_compatibility,
};
use super::vector::{
    LogicalVectorIndex, VectorIndexParams, build_vector_model_segment, fresh_vector_segment_params,
};
use super::{
    DatasetIndexExt, DatasetIndexInternalExt, load_all_indices, optimizable_indices,
    retain_committed_inverted_files, segment_has_inverted_details, segment_has_merge_primitive,
    segment_has_vector_details, validate_segment_metadata, vector_index_details,
};
use crate::dataset::Dataset;
use crate::dataset::transaction::{Operation, TransactionBuilder};
use crate::{Error, Result};

/// A plan to optimize the indices of a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeIndicesPlan {
    /// The dataset version the plan was made from.
    pub read_version: u64,
    /// The build units. Each writes one new segment.
    pub tasks: Vec<OptimizeIndexTask>,
    /// The merge units. Each yields one segment the commit publishes.
    pub merges: Vec<OptimizeIndexMerge>,
}

/// Build one new, uncommitted segment over a group of fragments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeIndexTask {
    pub read_version: u64,
    /// The merge this task's result feeds.
    pub group: u32,
    /// The fragments to index, by id in the dataset at `read_version`.
    pub fragments: Vec<u32>,
    /// The segment whose model and parameters the new segment copies: the
    /// index's newest committed segment, or, for a split rebuild, a
    /// model-only segment the plan trained that lives outside the manifest.
    #[serde(with = "index_metadata_hex")]
    pub reference: IndexMetadata,
    /// The committed segments this task may fold into its output. For an
    /// unsplit index these are all of its segments; for a split one, the
    /// segments its merge absorbs when this task is the merge's only input.
    #[serde(with = "index_metadata_hex::vec")]
    pub old_segments: Vec<IndexMetadata>,
    /// How many of `old_segments` to fold in, as
    /// [`OptimizeOptions::num_indices_to_merge`] counts them.
    pub num_indices_to_merge: Option<usize>,
    /// Rebuild the index from a freshly trained model.
    pub retrain: bool,
}

/// Combine the results of one merge group into the segment to commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeIndexMerge {
    pub read_version: u64,
    /// The key task results carry to reach this merge.
    pub group: u32,
    /// Committed segments merged into the output, which the commit then
    /// removes.
    #[serde(with = "index_metadata_hex::vec")]
    pub old_segments: Vec<IndexMetadata>,
    /// Committed segments the output supersedes without merging them: a
    /// rebuild replaces every old segment, and a segment whose fragments are
    /// all gone goes with whatever the index writes.
    pub replaces: Vec<Uuid>,
}

/// A segment produced by a task or a merge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexSegmentResult {
    pub read_version: u64,
    pub group: u32,
    /// The segment's metadata; its index name is `segment.name`.
    #[serde(with = "index_metadata_hex")]
    pub segment: IndexMetadata,
    /// The committed segments this segment supersedes; the commit removes them.
    pub replaces: Vec<Uuid>,
}

/// `IndexMetadata` as the hex of its manifest protobuf, which round-trips
/// every field (`created_at` at millisecond precision).
mod index_metadata_hex {
    use lance_table::format::{IndexMetadata, pb};
    use prost::Message;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    fn encode(metadata: &IndexMetadata) -> String {
        hex::encode(pb::IndexMetadata::from(metadata).encode_to_vec())
    }

    fn decode<E: serde::de::Error>(encoded: &str) -> Result<IndexMetadata, E> {
        let bytes = hex::decode(encoded).map_err(E::custom)?;
        let proto = pb::IndexMetadata::decode(bytes.as_slice()).map_err(E::custom)?;
        IndexMetadata::try_from(proto).map_err(E::custom)
    }

    pub fn serialize<S: Serializer>(
        metadata: &IndexMetadata,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        encode(metadata).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<IndexMetadata, D::Error> {
        decode(&String::deserialize(deserializer)?)
    }

    pub mod vec {
        use super::*;

        pub fn serialize<S: Serializer>(
            metadata: &[IndexMetadata],
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            metadata
                .iter()
                .map(encode)
                .collect::<Vec<_>>()
                .serialize(serializer)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Vec<IndexMetadata>, D::Error> {
            Vec::<String>::deserialize(deserializer)?
                .iter()
                .map(|encoded| decode(encoded))
                .collect()
        }
    }
}

/// The dataset at `version`: the handle itself when it is already there.
async fn dataset_at_version(dataset: &Dataset, version: u64) -> Result<Dataset> {
    if dataset.manifest.version == version {
        Ok(dataset.clone())
    } else {
        dataset.checkout_version(version).await
    }
}

/// The field an index is keyed on.
fn keyed_field<'a>(
    dataset: &'a Dataset,
    metadata: &IndexMetadata,
) -> Result<&'a lance_core::datatypes::Field> {
    let field_id = *metadata
        .fields
        .first()
        .ok_or_else(|| Error::index(format!("segment {} is missing field ids", metadata.uuid)))?;
    dataset
        .schema()
        .field_by_id(field_id)
        .ok_or_else(|| Error::index(format!("column {field_id} does not exist")))
}

/// Whether rewriting `segment` may shed stale postings. Under stable row ids
/// an update deletes a row's old copy and keeps its row id, so a scalar
/// segment keeps the old value's postings until a rewrite filters them out;
/// a deletion file on a covered fragment gives that away, conservatively.
fn may_hold_stale_postings(dataset: &Dataset, segment: &IndexMetadata) -> bool {
    if !dataset.manifest.uses_stable_row_ids() || segment_has_vector_details(segment) {
        return false;
    }
    let Some(covered) = segment.effective_fragment_bitmap(&dataset.fragment_bitmap) else {
        return false;
    };
    dataset
        .fragments()
        .iter()
        .any(|fragment| fragment.deletion_file.is_some() && covered.contains(fragment.id as u32))
}

async fn fragment_row_count(dataset: &Dataset, fragment: &Fragment) -> Result<u64> {
    match fragment.num_rows() {
        Some(rows) => Ok(rows as u64),
        None => Ok(dataset
            .count_rows_in_fragments(&[fragment.id as u32])
            .await? as u64),
    }
}

/// First-fit bin packing: each item goes into the first bin with room, or a
/// new one, so an item over the limit fills a bin alone. Without a limit
/// everything shares one bin.
fn pack_first_fit<T>(items: Vec<(T, u64)>, limit: Option<u64>) -> Vec<Vec<T>> {
    let limit = limit.unwrap_or(u64::MAX);
    // (items, rows used)
    let mut bins: Vec<(Vec<T>, u64)> = Vec::new();
    for (item, rows) in items {
        match bins
            .iter_mut()
            .find(|(_, used)| used.saturating_add(rows) <= limit)
        {
            Some((bin, used)) => {
                bin.push(item);
                *used = used.saturating_add(rows);
            }
            None => bins.push((vec![item], rows)),
        }
    }
    bins.into_iter().map(|(items, _)| items).collect()
}

/// What a split build decided for one index.
struct SplitInputs {
    /// The segment every new segment of this index is modeled on.
    reference: IndexMetadata,
    /// The fragments to index.
    fragments: Vec<Fragment>,
    /// The committed segments to absorb into the outputs.
    absorbed: Vec<IndexMetadata>,
    /// The committed segments the outputs supersede without absorbing them.
    replaces: Vec<Uuid>,
}

/// Skip an index a split build cannot rewrite, or refuse it when the caller
/// asked for it by name.
fn skip_unless_named(name: &str, named: bool, reason: &str) -> Result<Option<SplitInputs>> {
    if named {
        return Err(Error::not_supported(format!(
            "Cannot optimize index '{name}' as a split build: {reason}"
        )));
    }
    log::warn!("Skipping index '{name}' in the split optimize plan: {reason}");
    Ok(None)
}

/// Plan the optimization of the dataset's indices.
///
/// See the module documentation for what the plan contains and which indices
/// it leaves alone.
pub async fn plan_optimize_indices(
    dataset: &Dataset,
    options: &OptimizeOptions,
) -> Result<OptimizeIndicesPlan> {
    let indices = load_all_indices(dataset).await?;
    Ok(plan_optimize_indices_or_skip(dataset, &indices, options)
        .await?
        .unwrap_or_else(|| OptimizeIndicesPlan {
            read_version: dataset.manifest.version,
            tasks: Vec::new(),
            merges: Vec::new(),
        }))
}

/// Whether the table carries a tagged fragment reuse history this build
/// cannot interpret. Under such a history the reader lists no user segment,
/// so every fragment looks unindexed and each optimize would rebuild the
/// whole table only to have the result excluded again. Trim, superseded
/// pruning and remap refuse such a history; optimize leaves the table alone
/// the same way.
async fn has_unsupported_frag_reuse_history(
    dataset: &Dataset,
    indices: &[IndexMetadata],
) -> Result<bool> {
    let Some(entry) = indices
        .iter()
        .find(|index| lance_table::system_index::frag_reuse::metadata::is_tagged(index))
    else {
        return Ok(false);
    };
    Ok(super::frag_reuse::decode_frag_reuse_ledger(dataset, entry)
        .await?
        .has_unsupported_transitions())
}

/// The plan over `indices` (every entry of the manifest), or `None` when the
/// table is to be left alone.
pub(crate) async fn plan_optimize_indices_or_skip(
    dataset: &Dataset,
    indices: &[IndexMetadata],
    options: &OptimizeOptions,
) -> Result<Option<OptimizeIndicesPlan>> {
    if has_unsupported_frag_reuse_history(dataset, indices).await? {
        log::warn!(
            "Skipping index optimization: the tagged fragment reuse history carries \
             transitions this build cannot interpret; upgrade to a newer version of Lance"
        );
        return Ok(None);
    }
    let tagged = indices
        .iter()
        .any(lance_table::system_index::frag_reuse::metadata::is_tagged);
    let named = options.index_names.is_some();
    let bounded = options.max_rows_per_task.is_some() || options.max_rows_per_segment.is_some();
    let mut plan = OptimizeIndicesPlan {
        read_version: dataset.manifest.version,
        tasks: Vec::new(),
        merges: Vec::new(),
    };
    for (name, segments) in optimizable_indices(dataset, indices, options)? {
        let last = *segments.last().expect("an index has at least one segment");
        let unindexed = dataset.unindexed_fragments(&name).await?;
        // Without an N:1 merge the single-process optimize rebuilds the
        // absorbed segments from scratch, and the only split that reproduces
        // it is none.
        let split = bounded && segment_has_merge_primitive(last);
        if !split {
            if bounded {
                log::warn!(
                    "Index '{name}': its kind has no segment merge, so it is optimized in one task"
                );
            }
            plan_unsplit_index(&mut plan, dataset, &name, &segments, unindexed, options).await?;
            continue;
        }
        if tagged {
            skip_unless_named(
                &name,
                named,
                "the table has a tagged fragment reuse history, which the split merge cannot \
                 translate; optimize without max_rows_per_task and max_rows_per_segment",
            )?;
            continue;
        }
        let (field_path, _) = index_field_path(dataset, last).await?;
        let inputs = if segment_has_vector_details(last) {
            plan_split_vector_index(
                dataset,
                &name,
                &field_path,
                &segments,
                unindexed,
                options,
                named,
            )
            .await?
        } else {
            plan_split_scalar_index(
                dataset,
                &name,
                &field_path,
                &segments,
                unindexed,
                options,
                named,
            )
            .await?
        };
        let Some(inputs) = inputs else { continue };
        if inputs.fragments.is_empty() && inputs.absorbed.len() <= 1 {
            continue;
        }
        pack_split_index(&mut plan, dataset, inputs, options).await?;
    }
    Ok(Some(plan))
}

/// One index as one task: the single-process optimize over every segment and
/// every unindexed fragment, with the caller's options, and a merge that
/// passes the result through.
async fn plan_unsplit_index(
    plan: &mut OptimizeIndicesPlan,
    dataset: &Dataset,
    name: &str,
    segments: &[&IndexMetadata],
    unindexed: Vec<Fragment>,
    options: &OptimizeOptions,
) -> Result<()> {
    if !unsplit_index_needs_rewrite(dataset, name, segments, &unindexed, options).await? {
        return Ok(());
    }
    let read_version = plan.read_version;
    let group = plan.merges.len() as u32;
    let last = *segments.last().expect("an index has at least one segment");
    plan.tasks.push(OptimizeIndexTask {
        read_version,
        group,
        fragments: unindexed
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect(),
        reference: last.clone(),
        old_segments: segments.iter().map(|segment| (*segment).clone()).collect(),
        num_indices_to_merge: options.num_indices_to_merge,
        retrain: options.retrain,
    });
    plan.merges.push(OptimizeIndexMerge {
        read_version,
        group,
        old_segments: Vec::new(),
        replaces: Vec::new(),
    });
    Ok(())
}

/// Whether the single-process optimize would do anything for this index.
///
/// With unindexed fragments it always does. Without, this mirrors the early
/// returns of `merge_indices_with_unindexed_frags`, so a steady-state table
/// yields an empty plan rather than tasks that produce nothing.
async fn unsplit_index_needs_rewrite(
    dataset: &Dataset,
    name: &str,
    segments: &[&IndexMetadata],
    unindexed: &[Fragment],
    options: &OptimizeOptions,
) -> Result<bool> {
    if !unindexed.is_empty() {
        return Ok(true);
    }
    let last = *segments.last().expect("an index has at least one segment");
    if !segment_has_vector_details(last) {
        return Ok(select_segments_to_merge(dataset, segments, options).len() >= 2);
    }
    if segments
        .iter()
        .any(|segment| is_definition_only_segment(segment))
    {
        return Ok(false);
    }
    let tagged_coverage = tagged_segment_coverage(dataset, segments, None).await?;
    let (live, _) = partition_dormant_vector_segments(dataset, segments, tagged_coverage.as_ref());
    if live.is_empty() {
        return Ok(false);
    }
    if options.retrain {
        return Ok(true);
    }
    match options.num_indices_to_merge {
        Some(0) => return Ok(false),
        Some(_) => return Ok(true),
        None => {}
    }
    // Steady state: only a partition rebalance is worth a rewrite.
    let (field_path, _) = index_field_path(dataset, last).await?;
    let logical_index = dataset
        .open_logical_vector_index_for_maintenance(&field_path, name)
        .await?;
    let live_index = LogicalVectorIndex::try_new(
        name.to_string(),
        field_path,
        logical_index
            .iter()
            .filter(|(metadata, _)| live.iter().any(|segment| segment.uuid == metadata.uuid))
            .map(|(metadata, index)| (metadata.clone(), index.clone()))
            .collect(),
    )?;
    let view = live_index.as_ivf()?;
    let compatibility = vector_segment_compatibility(&view, "optimizing logical vector index")?;
    Ok(select_steady_state_rebalance(&view, compatibility)?.is_some())
}

#[allow(clippy::too_many_arguments)]
async fn plan_split_vector_index(
    dataset: &Dataset,
    name: &str,
    field_path: &str,
    segments: &[&IndexMetadata],
    fragments: Vec<Fragment>,
    options: &OptimizeOptions,
    named: bool,
) -> Result<Option<SplitInputs>> {
    let last = *segments.last().expect("an index has at least one segment");
    // A definition has no file to open and nothing to append to: train it,
    // from the parameters the definition carries.
    if segments
        .iter()
        .any(|segment| is_definition_only_segment(segment))
    {
        let params = last
            .index_details
            .as_deref()
            .and_then(vector_params_from_details)
            .ok_or_else(|| {
                Error::index(format!(
                    "index '{name}' awaits training but carries no parameters"
                ))
            })?;
        return plan_split_rebuild(dataset, name, field_path, segments, params, options, named)
            .await;
    }

    let logical_index = dataset
        .open_logical_vector_index_for_maintenance(field_path, name)
        .await?;
    // A dormant segment (its fragments all gone) holds nothing to append to,
    // and the single-process optimize drops it along the way. The newest live
    // segment is the reference; with none, the index is rebuilt as a retrain
    // is.
    let (live, dormant) = partition_dormant_vector_segments(dataset, segments, None);
    let reference = match live.last() {
        Some(reference) if !options.retrain => *reference,
        _ => {
            let (metadata, index) = logical_index.iter().last().ok_or_else(|| {
                Error::index(format!(
                    "logical vector index '{name}' has no physical segments"
                ))
            })?;
            let params = fresh_vector_segment_params(metadata, index.as_ref())?;
            return plan_split_rebuild(dataset, name, field_path, segments, params, options, named)
                .await;
        }
    };
    if logical_index.iter().any(|(metadata, index)| {
        metadata.uuid == reference.uuid && index.as_any().is::<LegacyIvfIndex>()
    }) {
        return skip_unless_named(
            name,
            named,
            "the index uses the legacy v1 file format, which the split merge cannot read; \
             set retrain to rebuild it in the current format",
        );
    }

    // Unset, `num_indices_to_merge` appends a delta, as the single-process
    // optimize does when no partition needs rebalancing.
    let num_to_merge = options
        .num_indices_to_merge
        .unwrap_or(0)
        .min(segments.len());
    let absorbed = segments[segments.len() - num_to_merge..]
        .iter()
        .map(|segment| (*segment).clone())
        .collect();
    let absorbed = absorbable(dataset, name, reference, absorbed).await?;
    if !absorbed.is_empty() {
        // Segments merge physically only when they share one model.
        let pairs = logical_index
            .iter()
            .filter(|(metadata, _)| {
                metadata.uuid == reference.uuid
                    || absorbed.iter().any(|segment| segment.uuid == metadata.uuid)
            })
            .map(|(metadata, index)| (metadata.clone(), index.clone()))
            .collect();
        let candidates =
            LogicalVectorIndex::try_new(name.to_string(), field_path.to_string(), pairs)?;
        let compatibility =
            vector_segment_compatibility(&candidates.as_ivf()?, "planning distributed optimize")?;
        if compatibility != VectorSegmentCompatibility::SharedModel {
            return Err(Error::index(format!(
                "Optimize index '{name}': the segments absorbed by num_indices_to_merge do \
                 not share one IVF model; leave num_indices_to_merge unset to append only, \
                 or set retrain to rebuild them"
            )));
        }
    }
    Ok(Some(SplitInputs {
        reference: reference.clone(),
        fragments,
        absorbed,
        replaces: dormant.iter().map(|segment| segment.uuid).collect(),
    }))
}

#[allow(clippy::too_many_arguments)]
async fn plan_split_scalar_index(
    dataset: &Dataset,
    name: &str,
    field_path: &str,
    segments: &[&IndexMetadata],
    fragments: Vec<Fragment>,
    options: &OptimizeOptions,
    named: bool,
) -> Result<Option<SplitInputs>> {
    let last = *segments.last().expect("an index has at least one segment");
    let reference_index = dataset
        .open_scalar_index_for_maintenance(field_path, &last.uuid, &NoOpMetricsCollector)
        .await?;
    if reference_index.update_criteria().requires_old_data {
        return skip_unless_named(
            name,
            named,
            "the index can only be rebuilt from the whole column; optimize without \
             max_rows_per_task and max_rows_per_segment, which also rebuilds it in the \
             current format",
        );
    }
    let selected: Vec<IndexMetadata> = select_segments_to_merge(dataset, segments, options)
        .into_iter()
        .cloned()
        .collect();
    // The single-process merge removes a selected segment whose fragments are
    // all gone along with the rest; the split merge cannot absorb it.
    let dead = selected
        .iter()
        .filter(|segment| {
            segment
                .effective_fragment_bitmap(&dataset.fragment_bitmap)
                .is_some_and(|bitmap| bitmap.is_empty())
        })
        .map(|segment| segment.uuid)
        .collect();
    let absorbed = absorbable(dataset, name, last, selected).await?;
    Ok(Some(SplitInputs {
        reference: last.clone(),
        fragments,
        absorbed,
        replaces: dead,
    }))
}

/// Train a fresh model over the whole table and plan every live fragment
/// against it; the commit then replaces every old segment.
#[allow(clippy::too_many_arguments)]
async fn plan_split_rebuild(
    dataset: &Dataset,
    name: &str,
    field_path: &str,
    segments: &[&IndexMetadata],
    params: VectorIndexParams,
    options: &OptimizeOptions,
    named: bool,
) -> Result<Option<SplitInputs>> {
    let last = *segments.last().expect("an index has at least one segment");
    if expects_definition_only(dataset, field_path, &params).await? {
        return skip_unless_named(
            name,
            named,
            "the column has too few rows to train the index",
        );
    }
    let uuid = Uuid::new_v4();
    let frag_reuse_index = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
    let files = build_vector_model_segment(
        dataset,
        field_path,
        name,
        uuid,
        &params,
        frag_reuse_index,
        options.progress.clone(),
    )
    .await?;
    let reference = IndexMetadata {
        uuid,
        name: name.to_string(),
        fields: last.fields.clone(),
        covering_fields: Vec::new(),
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(RoaringBitmap::new()),
        index_details: Some(Arc::new(vector_index_details(&params))),
        index_version: params.index_type().version(),
        created_at: Some(Utc::now()),
        base_id: None,
        files: Some(files),
    };
    Ok(Some(SplitInputs {
        reference,
        fragments: dataset.fragments().as_ref().clone(),
        absorbed: Vec::new(),
        replaces: segments.iter().map(|segment| segment.uuid).collect(),
    }))
}

/// Drop the candidates a split merge cannot absorb. A segment with no live
/// coverage has nothing to contribute; any other problem means the index only
/// gets its new data appended, with a warning.
async fn absorbable(
    dataset: &Dataset,
    name: &str,
    reference: &IndexMetadata,
    mut absorbed: Vec<IndexMetadata>,
) -> Result<Vec<IndexMetadata>> {
    absorbed.retain(|segment| {
        segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .is_some_and(|bitmap| !bitmap.is_empty())
    });
    let reason = if absorbed
        .iter()
        .any(|segment| segment.index_version != reference.index_version)
    {
        Some("their on-disk index versions differ")
    } else if absorbed.iter().any(|segment| segment.base_id.is_some()) {
        Some("a segment lives in the base dataset of a shallow clone")
    } else if let Some(frag_reuse_index) =
        dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?
        && fragment_reuse_affects_segments(&frag_reuse_index, absorbed.iter())
    {
        Some("a segment has a pending compaction remap")
    } else {
        None
    };
    if let Some(reason) = reason {
        log::warn!(
            "Index '{name}': appending new segments without merging existing ones because {reason}"
        );
        absorbed.clear();
    }
    Ok(absorbed)
}

/// Pack one split index's inputs into tasks and merges and append them to
/// the plan.
async fn pack_split_index(
    plan: &mut OptimizeIndicesPlan,
    dataset: &Dataset,
    inputs: SplitInputs,
    options: &OptimizeOptions,
) -> Result<()> {
    let SplitInputs {
        reference,
        fragments,
        absorbed,
        replaces,
    } = inputs;
    // A scalar segment absorbed to shed stale postings is rewritten even
    // alone, but only alongside new data, as the single-process optimize
    // does: the deletion files outlive the rewrite, so a segment too large to
    // join a merge would otherwise be rewritten every time.
    let needs_lone_rewrite = |segment: &IndexMetadata| {
        !fragments.is_empty() && may_hold_stale_postings(dataset, segment)
    };
    let mut rows_by_fragment = HashMap::new();
    for fragment in dataset.fragments().iter() {
        rows_by_fragment.insert(
            fragment.id as u32,
            fragment_row_count(dataset, fragment).await?,
        );
    }

    let build_inputs = fragments
        .iter()
        .map(|fragment| {
            let id = fragment.id as u32;
            (id, rows_by_fragment.get(&id).copied().unwrap_or(0))
        })
        .collect();
    // A task's output is never split, so it holds no more than a segment may.
    let task_limit = [options.max_rows_per_task, options.max_rows_per_segment]
        .into_iter()
        .flatten()
        .min();
    let fragment_groups = pack_first_fit(build_inputs, task_limit);

    enum MergeInput {
        Absorbed(IndexMetadata),
        Build(Vec<u32>),
    }
    let mut merge_inputs = Vec::with_capacity(absorbed.len() + fragment_groups.len());
    for segment in absorbed {
        let rows = segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .unwrap_or_default()
            .iter()
            .map(|id| rows_by_fragment.get(&id).copied().unwrap_or(0))
            .sum();
        merge_inputs.push((MergeInput::Absorbed(segment), rows));
    }
    for group in fragment_groups {
        let rows = group
            .iter()
            .map(|id| rows_by_fragment.get(id).copied().unwrap_or(0))
            .sum();
        merge_inputs.push((MergeInput::Build(group), rows));
    }

    for bin in pack_first_fit(merge_inputs, options.max_rows_per_segment) {
        let mut absorbed = Vec::new();
        let mut builds = Vec::new();
        for input in bin {
            match input {
                MergeInput::Absorbed(segment) => absorbed.push(segment),
                MergeInput::Build(fragments) => builds.push(fragments),
            }
        }
        // One old segment and nothing to add to it: leave it as it is.
        if builds.is_empty()
            && absorbed.len() <= 1
            && !absorbed.first().is_some_and(needs_lone_rewrite)
        {
            continue;
        }
        let read_version = plan.read_version;
        let group = plan.merges.len() as u32;
        // A single task folds the old segments in itself, in one pass.
        let (absorbed_by_task, absorbed_by_merge) = if builds.len() == 1 {
            (absorbed, Vec::new())
        } else {
            (Vec::new(), absorbed)
        };
        for fragments in builds {
            plan.tasks.push(OptimizeIndexTask {
                read_version,
                group,
                fragments,
                reference: reference.clone(),
                num_indices_to_merge: Some(absorbed_by_task.len()),
                old_segments: absorbed_by_task.clone(),
                retrain: false,
            });
        }
        plan.merges.push(OptimizeIndexMerge {
            read_version,
            group,
            old_segments: absorbed_by_merge,
            replaces: replaces.clone(),
        });
    }
    Ok(())
}

impl OptimizeIndexTask {
    /// Build this task's segment. `dataset` may be at any version; the task
    /// reads the version it was planned from. `None` when there was nothing
    /// to do after all (a steady-state table, say).
    pub async fn execute(&self, dataset: &Dataset) -> Result<Option<IndexSegmentResult>> {
        self.execute_with_progress(dataset, noop_progress()).await
    }

    /// [`Self::execute`], reporting the build's stages to `progress`.
    pub(crate) async fn execute_with_progress(
        &self,
        dataset: &Dataset,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<Option<IndexSegmentResult>> {
        let dataset = Arc::new(dataset_at_version(dataset, self.read_version).await?);
        let fragments = dataset
            .get_fragments_from_ids(&self.fragments)?
            .into_iter()
            .map(|fragment| fragment.metadata().clone())
            .collect::<Vec<_>>();
        let mut options = OptimizeOptions::default().progress(progress);
        options.num_indices_to_merge = self.num_indices_to_merge;
        options.retrain = self.retrain;
        let results = if self.old_segments.is_empty() && segment_has_vector_details(&self.reference)
        {
            let field = keyed_field(&dataset, &self.reference)?;
            let field_path = dataset.schema().field_path(field.id)?;
            let index = dataset
                .open_vector_index_from_metadata_with_plan(
                    &field_path,
                    &self.reference,
                    Some(&SegmentRemappingPlan::Identity),
                    OpenPurpose::Maintenance,
                    &NoOpMetricsCollector,
                )
                .await?;
            Some(
                append_vector_segment(
                    &dataset,
                    &field_path,
                    field.nullable,
                    &self.reference,
                    &index,
                    &fragments,
                    &options,
                )
                .await?,
            )
        } else {
            // The single-process optimize over this task's fragments.
            let candidates: Vec<&IndexMetadata> = if self.old_segments.is_empty() {
                vec![&self.reference]
            } else {
                self.old_segments.iter().collect()
            };
            merge_indices_with_unindexed_frags(dataset.clone(), &candidates, &fragments, &options)
                .await?
        };
        Ok(results.map(|results| IndexSegmentResult {
            read_version: self.read_version,
            group: self.group,
            replaces: results
                .removed_indices
                .iter()
                .map(|segment| segment.uuid)
                .collect(),
            segment: results.into_metadata(&self.reference),
        }))
    }
}

impl OptimizeIndexMerge {
    /// Merge this group's inputs: the old segments it lists plus the task
    /// results that carry its `group`. A lone task result passes through
    /// unchanged, and a lone old segment is rewritten only if it may hold
    /// stale postings; with no inputs (every task of the group produced
    /// nothing) there is no result.
    pub async fn execute(
        &self,
        dataset: &Dataset,
        task_results: &[IndexSegmentResult],
    ) -> Result<Option<IndexSegmentResult>> {
        self.execute_with_progress(dataset, task_results, noop_progress())
            .await
    }

    /// [`Self::execute`], reporting the merge's stages to `progress`.
    pub(crate) async fn execute_with_progress(
        &self,
        dataset: &Dataset,
        task_results: &[IndexSegmentResult],
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<Option<IndexSegmentResult>> {
        let mut inputs = self.old_segments.clone();
        let mut from_tasks = 0;
        let mut replaces: Vec<Uuid> = self
            .old_segments
            .iter()
            .map(|segment| segment.uuid)
            .chain(self.replaces.iter().copied())
            .collect();
        for result in task_results
            .iter()
            .filter(|result| result.group == self.group)
        {
            if result.read_version != self.read_version {
                return Err(Error::invalid_input(format!(
                    "task result for group {} was built from dataset version {}, but this merge \
                     was planned from version {}",
                    self.group, result.read_version, self.read_version
                )));
            }
            from_tasks += 1;
            inputs.push(result.segment.clone());
            replaces.extend(result.replaces.iter().copied());
        }
        let segment = match inputs.len() {
            0 => return Ok(None),
            // A lone task result needs no work. A lone old segment (planned
            // alone, or its tasks produced nothing) is rewritten only to shed
            // stale postings; otherwise it stays as it is, since passing it
            // through would have the commit remove it.
            1 if from_tasks == 1 => inputs.pop().expect("one input"),
            1 if segment_has_vector_details(&inputs[0]) => return Ok(None),
            _ => {
                let dataset = dataset_at_version(dataset, self.read_version).await?;
                if inputs.len() == 1 && !may_hold_stale_postings(&dataset, &inputs[0]) {
                    return Ok(None);
                }
                if segment_has_vector_details(&inputs[0]) {
                    let field = keyed_field(&dataset, &inputs[0])?;
                    merge_vector_segments(&dataset, field.id, inputs, progress).await?
                } else {
                    dataset.merge_existing_index_segments(inputs).await?
                }
            }
        };
        Ok(Some(IndexSegmentResult {
            read_version: self.read_version,
            group: self.group,
            segment,
            replaces,
        }))
    }
}

/// Merge vector segments the way the single-process optimize does: the index
/// builder copies every input's partitions into one new segment, keeping only
/// rows whose fragments are still live. The inputs are opened from their
/// metadata, so uncommitted task output merges like a committed segment.
async fn merge_vector_segments(
    dataset: &Dataset,
    field_id: i32,
    inputs: Vec<IndexMetadata>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<IndexMetadata> {
    let last = inputs.last().expect("at least two inputs");
    // A task result passed twice (a retried task, say) would index its rows
    // twice; the scalar merge refuses overlapping inputs the same way.
    validate_segment_metadata(&last.name, &inputs)?;
    let field_path = dataset.schema().field_path(field_id)?;
    let mut pairs = Vec::with_capacity(inputs.len());
    for metadata in &inputs {
        let index = dataset
            .open_vector_index_from_metadata_with_plan(
                &field_path,
                metadata,
                Some(&SegmentRemappingPlan::Identity),
                OpenPurpose::Maintenance,
                &NoOpMetricsCollector,
            )
            .await?;
        pairs.push((metadata.clone(), index));
    }
    let logical_index = LogicalVectorIndex::try_new(last.name.clone(), field_path.clone(), pairs)?;
    let (uuid, merged_count, files) = optimize_vector_indices(
        dataset.clone(),
        Option::<
            lance_io::stream::RecordBatchStreamAdapter<
                futures::stream::Empty<Result<arrow_array::RecordBatch>>,
            >,
        >::None,
        &field_path,
        &logical_index.as_ivf()?,
        &OptimizeOptions::merge(inputs.len()).progress(progress),
    )
    .boxed()
    .await?;
    if merged_count != inputs.len() {
        return Err(Error::internal(format!(
            "optimize merge combined {merged_count} segments, expected {}",
            inputs.len()
        )));
    }
    Ok(IndexMetadata {
        uuid,
        name: last.name.clone(),
        fields: last.fields.clone(),
        covering_fields: last.covering_fields.clone(),
        dataset_version: inputs
            .iter()
            .map(|metadata| metadata.dataset_version)
            .min()
            .unwrap_or(dataset.manifest.version),
        fragment_bitmap: Some(
            inputs
                .iter()
                .filter_map(|metadata| metadata.fragment_bitmap.clone())
                .fold(RoaringBitmap::new(), |acc, bitmap| acc | bitmap),
        ),
        index_details: last
            .index_details
            .clone()
            .or_else(|| Some(Arc::new(super::vector_index_details_default()))),
        index_version: last.index_version,
        created_at: Some(Utc::now()),
        base_id: None,
        files: Some(files),
    })
}

/// Commit the merged segments in one `CreateIndex` transaction.
///
/// The segments the results claim to replace are removed, exactly as the
/// single-process optimize removes what its merge reports. Fragments those
/// segments covered that no result covers (a rebuild missing results) go back
/// to unindexed, and the next optimize indexes them.
///
/// Without results nothing is committed, unless the table requires MemWAL
/// catch-up: coverage is derived at commit time, so an index that already
/// spans the table records its position only if there is a commit to record
/// it on -- the ordinary case after a remap or a compaction that advanced a
/// generation without changing fragments. Returning early there would leave
/// the position missing forever and the repair rescheduling itself.
pub async fn commit_optimize_indices(
    dataset: &mut Dataset,
    results: Vec<IndexSegmentResult>,
    options: &OptimizeOptions,
) -> Result<()> {
    if results.is_empty() {
        let indices = load_all_indices(dataset).await?;
        if !dataset.mem_wal_catch_up_would_advance(&indices)?
            || has_unsupported_frag_reuse_history(dataset, &indices).await?
        {
            return Ok(());
        }
        let transaction = TransactionBuilder::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![],
                removed_indices: vec![],
            },
        )
        .transaction_properties(options.transaction_properties.clone())
        .build();
        return dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await;
    }
    let read_version = results[0].read_version;
    if results
        .iter()
        .any(|result| result.read_version != read_version)
    {
        return Err(Error::invalid_input(
            "results to commit were planned from different dataset versions".to_string(),
        ));
    }
    let planned = dataset_at_version(dataset, read_version).await?;
    let committed_segments = load_all_indices(&planned).await?;

    let mut replaced: HashSet<Uuid> = HashSet::new();
    let mut by_name: BTreeMap<String, Vec<IndexMetadata>> = BTreeMap::new();
    for result in results {
        // A result only ever replaces segments of its own index.
        let name = &result.segment.name;
        if let Some(uuid) = result.replaces.iter().find(|uuid| {
            !committed_segments
                .iter()
                .any(|segment| segment.uuid == **uuid && segment.name == *name)
        }) {
            return Err(Error::invalid_input(format!(
                "result for index '{name}' replaces segment {uuid}, which that index does not \
                 have at dataset version {read_version}"
            )));
        }
        replaced.extend(result.replaces);
        by_name
            .entry(result.segment.name.clone())
            .or_default()
            .push(result.segment);
    }

    let mut new_indices = Vec::new();
    for (name, mut segments) in by_name {
        validate_segment_metadata(&name, &segments)?;
        let incoming = segments
            .iter()
            .filter_map(|segment| segment.fragment_bitmap.clone())
            .fold(RoaringBitmap::new(), |acc, bitmap| acc | bitmap);
        // A retained segment and a new one must not both answer for a fragment.
        for old in committed_segments
            .iter()
            .filter(|segment| segment.name == name && !replaced.contains(&segment.uuid))
        {
            let effective = old
                .effective_fragment_bitmap(&planned.fragment_bitmap)
                .unwrap_or_default();
            if !effective.is_disjoint(&incoming) {
                return Err(Error::internal(format!(
                    "optimize of index '{name}' produced a segment overlapping segment {}, \
                     which it does not replace",
                    old.uuid
                )));
            }
        }
        for segment in &mut segments {
            if segment_has_inverted_details(segment)
                && let Some(files) = segment.files.as_mut()
            {
                retain_committed_inverted_files(files);
            }
        }
        new_indices.extend(segments);
    }
    let removed_indices = committed_segments
        .iter()
        .filter(|segment| replaced.contains(&segment.uuid))
        .cloned()
        .collect();

    let transaction = TransactionBuilder::new(
        read_version,
        Operation::CreateIndex {
            new_indices,
            removed_indices,
        },
    )
    .transaction_properties(options.transaction_properties.clone())
    .build();
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_index::vector::{ivf::IvfBuildParams, pq::PQBuildParams};
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;

    use crate::dataset::WriteParams;

    const DIM: usize = 64;
    const VECTOR_INDEX: &str = "vector_idx";
    const SCALAR_INDEX: &str = "id_idx";

    /// `rows` rows of `id` (from `first_id`) and a random `vector`.
    fn batch(first_id: i32, rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            ),
        ]));
        let ids = Int32Array::from_iter_values(first_id..first_id + rows as i32);
        let vectors =
            FixedSizeListArray::try_new_from_values(generate_random_array(rows * DIM), DIM as i32)
                .unwrap();
        RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(vectors)]).unwrap()
    }

    fn params(rows_per_fragment: usize) -> WriteParams {
        WriteParams {
            max_rows_per_file: rows_per_fragment,
            ..Default::default()
        }
    }

    /// A dataset of `rows` rows in fragments of `rows_per_fragment`.
    async fn write_dataset(uri: &str, rows: usize, rows_per_fragment: usize) -> Dataset {
        let batch = batch(0, rows);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        Dataset::write(reader, uri, Some(params(rows_per_fragment)))
            .await
            .unwrap()
    }

    /// Append `rows` rows in fragments of `rows_per_fragment`.
    async fn append(dataset: &mut Dataset, rows: usize, rows_per_fragment: usize) {
        let first_id = dataset.count_rows(None).await.unwrap() as i32;
        let batch = batch(first_id, rows);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        dataset
            .append(reader, Some(params(rows_per_fragment)))
            .await
            .unwrap();
    }

    async fn create_vector_index(dataset: &mut Dataset) {
        let params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            IvfBuildParams::new(2),
            PQBuildParams {
                num_sub_vectors: 2,
                ..Default::default()
            },
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(VECTOR_INDEX.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();
    }

    async fn create_btree_index(dataset: &mut Dataset) {
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some(SCALAR_INDEX.to_string()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                true,
            )
            .await
            .unwrap();
    }

    /// Execute every task, then every merge, as a driver would.
    async fn execute_plan(
        dataset: &Dataset,
        plan: &OptimizeIndicesPlan,
    ) -> Vec<IndexSegmentResult> {
        let mut built = Vec::new();
        for task in &plan.tasks {
            built.extend(task.execute(dataset).await.unwrap());
        }
        let mut merged = Vec::new();
        for merge in &plan.merges {
            merged.extend(merge.execute(dataset, &built).await.unwrap());
        }
        merged
    }

    async fn segments(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
        dataset.load_indices_by_name(name).await.unwrap()
    }

    fn coverage(segments: &[IndexMetadata]) -> RoaringBitmap {
        segments
            .iter()
            .filter_map(|segment| segment.fragment_bitmap.clone())
            .fold(RoaringBitmap::new(), |acc, bitmap| acc | bitmap)
    }

    fn task_groups(plan: &OptimizeIndicesPlan) -> Vec<(u32, Vec<u32>)> {
        plan.tasks
            .iter()
            .map(|task| (task.group, task.fragments.clone()))
            .collect()
    }

    async fn indexed_rows(dataset: &Dataset, segments: &[IndexMetadata]) -> u64 {
        let mut rows = 0;
        for segment in segments {
            let index = dataset
                .open_vector_index("vector", &segment.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            rows += index.num_rows();
        }
        rows
    }

    /// The value survives serialization: decoding it and encoding it again
    /// yields the same JSON.
    fn assert_round_trip<T: Serialize + for<'de> Deserialize<'de>>(value: &T) {
        let json = serde_json::to_string(value).unwrap();
        let decoded: T = serde_json::from_str(&json).unwrap();
        assert_eq!(serde_json::to_string(&decoded).unwrap(), json);
    }

    #[test]
    fn test_pack_first_fit() {
        // Over the limit: a bin of its own. Otherwise first bin with room.
        let bins = pack_first_fit(
            vec![("a", 300), ("b", 5000), ("c", 400), ("d", 200), ("e", 900)],
            Some(1000),
        );
        assert_eq!(bins, vec![vec!["a", "c", "d"], vec!["b"], vec!["e"]]);
        assert_eq!(
            pack_first_fit(vec![("a", 1), ("b", 2)], None),
            vec![vec!["a", "b"]]
        );
        assert!(pack_first_fit::<&str>(vec![], None).is_empty());
    }

    #[tokio::test]
    async fn test_vector_append_then_merge() {
        let test_dir = TempStrDir::default();
        let mut dataset = write_dataset(&test_dir, 1000, 500).await;
        create_vector_index(&mut dataset).await;
        let original = segments(&dataset, VECTOR_INDEX).await[0].uuid;
        append(&mut dataset, 1000, 250).await; // fragments 2..=5

        // A task's output is never split, so the segment bound bounds tasks too.
        let options = OptimizeOptions::append().max_rows_per_segment(Some(500));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(task_groups(&plan), vec![(0, vec![2, 3]), (1, vec![4, 5])]);

        let options = options.max_rows_per_task(Some(250));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(
            task_groups(&plan),
            vec![(0, vec![2]), (0, vec![3]), (1, vec![4]), (1, vec![5])]
        );
        assert!(
            plan.tasks
                .iter()
                .all(|task| task.reference.uuid == original && task.old_segments.is_empty())
        );
        assert_round_trip(&plan);

        // The workers see a newer version than the plan was made from.
        append(&mut dataset, 250, 250).await; // fragment 6
        let before_commit = dataset.manifest.version;
        let mut built = Vec::new();
        for task in &plan.tasks {
            let result = task.execute(&dataset).await.unwrap().unwrap();
            assert_round_trip(&result);
            assert_eq!(result.read_version, plan.read_version);
            built.push(result);
        }
        let mut merged = Vec::new();
        for merge in &plan.merges {
            merged.push(merge.execute(&dataset, &built).await.unwrap().unwrap());
        }
        assert_eq!(
            merged
                .iter()
                .map(|result| result.segment.fragment_bitmap.clone().unwrap())
                .collect::<Vec<_>>(),
            vec![
                RoaringBitmap::from_iter([2u32, 3]),
                RoaringBitmap::from_iter([4u32, 5])
            ]
        );

        commit_optimize_indices(&mut dataset, merged, &options)
            .await
            .unwrap();
        assert_eq!(dataset.manifest.version, before_commit + 1);
        let segments = segments(&dataset, VECTOR_INDEX).await;
        assert_eq!(segments.len(), 3);
        assert_eq!(coverage(&segments), RoaringBitmap::from_iter(0u32..=5));
        assert_eq!(indexed_rows(&dataset, &segments).await, 2000);

        // Only the fragment appended after the plan is left to index.
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(task_groups(&plan), vec![(0, vec![6])]);
    }

    #[tokio::test]
    async fn test_vector_folds_old_segments_in_merge() {
        let test_dir = TempStrDir::default();
        let mut dataset = write_dataset(&test_dir, 1000, 500).await;
        create_vector_index(&mut dataset).await;
        append(&mut dataset, 250, 250).await; // fragment 2
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let old_uuids: HashSet<Uuid> = segments(&dataset, VECTOR_INDEX)
            .await
            .iter()
            .map(|segment| segment.uuid)
            .collect();
        assert_eq!(old_uuids.len(), 2);
        // Retire fragment 1 entirely: the first segment's stored coverage now
        // names a fragment that no longer exists.
        dataset.delete("id >= 500 AND id < 1000").await.unwrap();
        append(&mut dataset, 500, 250).await; // fragments 3, 4

        // With one task for the new fragments, that task folds the old
        // segments in itself and its merge only passes the result through.
        let options = OptimizeOptions::merge(2).max_rows_per_task(Some(500));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(task_groups(&plan), vec![(0, vec![3, 4])]);
        assert_eq!(plan.tasks[0].old_segments.len(), 2);
        assert!(plan.merges[0].old_segments.is_empty());

        let options = OptimizeOptions::merge(2).max_rows_per_task(Some(250));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks.len(), 2);
        assert!(plan.tasks.iter().all(|task| task.old_segments.is_empty()));
        assert_eq!(plan.merges.len(), 1);
        assert_eq!(plan.merges[0].old_segments.len(), 2);

        let merged = execute_plan(&dataset, &plan).await;
        commit_optimize_indices(&mut dataset, merged, &options)
            .await
            .unwrap();
        let segments = segments(&dataset, VECTOR_INDEX).await;
        assert_eq!(segments.len(), 1);
        assert!(!old_uuids.contains(&segments[0].uuid));
        assert_eq!(
            coverage(&segments) & dataset.fragment_bitmap.as_ref(),
            *dataset.fragment_bitmap.as_ref()
        );
        // The rows of the retired fragment were filtered out of the merge.
        assert_eq!(indexed_rows(&dataset, &segments).await, 1250);
    }

    #[tokio::test]
    async fn test_vector_retrain() {
        let test_dir = TempStrDir::default();
        let mut dataset = write_dataset(&test_dir, 1000, 500).await;
        create_vector_index(&mut dataset).await;
        let original = segments(&dataset, VECTOR_INDEX).await[0].uuid;
        append(&mut dataset, 500, 250).await; // fragments 2, 3

        let options = OptimizeOptions::retrain()
            .max_rows_per_task(Some(500))
            .max_rows_per_segment(Some(500));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(
            task_groups(&plan),
            vec![(0, vec![0]), (1, vec![1]), (2, vec![2, 3])]
        );
        let model = &plan.tasks[0].reference;
        assert_ne!(model.uuid, original);
        assert_eq!(model.fragment_bitmap, Some(RoaringBitmap::new()));
        assert!(
            plan.tasks
                .iter()
                .all(|task| task.reference == *model && task.old_segments.is_empty())
        );

        let merged = execute_plan(&dataset, &plan).await;
        // Committing only fragment 0's result still replaces the old segment;
        // fragments 1..=3 go back to unindexed and a later append indexes them
        // against the new model.
        commit_optimize_indices(&mut dataset, vec![merged[0].clone()], &options)
            .await
            .unwrap();
        let rebuilt = segments(&dataset, VECTOR_INDEX).await;
        assert_eq!(rebuilt.len(), 1);
        assert_ne!(rebuilt[0].uuid, original);
        assert_eq!(coverage(&rebuilt), RoaringBitmap::from_iter([0u32]));

        // The rest is appended against the rebuilt segment.
        let options = OptimizeOptions::append()
            .max_rows_per_task(Some(500))
            .max_rows_per_segment(Some(500));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(task_groups(&plan), vec![(0, vec![1]), (1, vec![2, 3])]);
        assert!(
            plan.tasks
                .iter()
                .all(|task| task.reference.uuid == rebuilt[0].uuid)
        );
    }

    #[tokio::test]
    async fn test_partial_results_keep_the_unmerged_vector_segment() {
        let test_dir = TempStrDir::default();
        let mut dataset = write_dataset(&test_dir, 1000, 500).await;
        create_vector_index(&mut dataset).await;
        create_btree_index(&mut dataset).await;
        let original = segments(&dataset, VECTOR_INDEX).await[0].uuid;
        append(&mut dataset, 500, 250).await; // fragments 2, 3

        // Each index's merge folds its old segment with two task results. The
        // vector tasks fail; the BTree ones succeed and get committed.
        let options = OptimizeOptions::merge(1).max_rows_per_task(Some(250));
        let mut plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        plan.tasks
            .retain(|task| task.reference.name != VECTOR_INDEX);
        let merged = execute_plan(&dataset, &plan).await;
        commit_optimize_indices(&mut dataset, merged, &options)
            .await
            .unwrap();
        let scalar = segments(&dataset, SCALAR_INDEX).await;
        assert_eq!(coverage(&scalar), RoaringBitmap::from_iter(0u32..=3));
        let vector = segments(&dataset, VECTOR_INDEX).await;
        assert_eq!(
            vector
                .iter()
                .map(|segment| segment.uuid)
                .collect::<Vec<_>>(),
            vec![original]
        );
    }

    #[tokio::test]
    async fn test_split_merge_drops_segments_with_no_live_fragments() {
        let test_dir = TempStrDir::default();
        let mut dataset = write_dataset(&test_dir, 1000, 500).await;
        create_btree_index(&mut dataset).await;
        for _ in 0..2 {
            append(&mut dataset, 500, 500).await; // fragments 2, then 3
            dataset
                .optimize_indices(&OptimizeOptions::append())
                .await
                .unwrap();
        }
        assert_eq!(segments(&dataset, SCALAR_INDEX).await.len(), 3);
        // The first segment's fragments are gone.
        dataset.delete("id < 1000").await.unwrap();

        // Nothing new to index: one merge folds the two live segments and
        // replaces the dead one.
        let options = OptimizeOptions::merge(3).max_rows_per_segment(Some(1000));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert!(plan.tasks.is_empty());
        assert_eq!(plan.merges.len(), 1);
        assert_eq!(plan.merges[0].old_segments.len(), 2);
        let merged = execute_plan(&dataset, &plan).await;
        commit_optimize_indices(&mut dataset, merged, &options)
            .await
            .unwrap();
        let segments = segments(&dataset, SCALAR_INDEX).await;
        assert_eq!(segments.len(), 1);
        assert_eq!(coverage(&segments), RoaringBitmap::from_iter([2u32, 3]));
    }
}
