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
//! Bitmap, NGram, Inverted, ZoneMap) is split: tasks build one uncommitted
//! segment each over a group of fragments, encoded against the model of a reference
//! segment; merges combine the task results that carry their `group` with the
//! committed segments they list into one segment. A task that is the only
//! input of its merge folds those old segments in itself, in one pass. For a
//! vector index `num_indices_to_merge` unset means append only: the
//! single-process optimize appends a delta too, unless the partitions need
//! rebalancing, which a split build does not do (set `retrain` for that). The
//! other kinds are rebuilt whole by the single-process optimize, so they stay
//! one task. ZoneMap reuses compatible v1 segments; v0 or incompatible segments
//! without pending remaps are rebuilt over their live coverage plus new fragments,
//! using the same task bounds. Pending remaps follow the shared policy: retain
//! old segments and build only new data. A fragment is indivisible even when it
//! exceeds the bound.
//!
//! [`OptimizeOptions::consolidate`] instead selects a suffix of complete vector
//! segments within a physical-row budget, leaving unindexed fragments alone.
//! Shared models merge directly; differing models are re-encoded in tasks under
//! the newest selected model, then merged into one output. A segment selection
//! ([`OptimizeOptions::consolidate_selected`]) instead trains a model on its own
//! rows when the models differ, and may pack its output into several segments
//! of at most `max_rows_per_segment` live rows. All selected coverage
//! must be present before committing. A caller may retry the same results on a
//! refreshed dataset after a concurrent append or delete; a changed source owner
//! or restored source rows require replanning. Tagged fragment reuse is not
//! supported in this mode.
//!
//! Planning is not always cheap: a split vector index that has to be rebuilt
//! (`retrain`, every segment's fragments gone, or nothing but a definition so
//! far) is trained by the planner, which writes the model as a segment outside
//! the manifest for the tasks to encode against; cleanup reclaims it like any
//! unreferenced index directory.
//!
//! The commit publishes every merged segment in one `CreateIndex` transaction
//! anchored at the plan's `read_version`, and removes the segments the results
//! replace. Consolidation may reanchor to the caller's refreshed version after
//! checking that the source data and ownership still permit reusing the output.
//! Outside consolidation, results may be partial: whatever they do not cover
//! stays unindexed for the next optimize.
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

mod consolidate;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arrow_schema::DataType;
use chrono::Utc;
use futures::FutureExt;
use itertools::Itertools;
use lance_index::is_system_index;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::{IndexBuildProgress, noop_progress};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::DatasetIndexExt;
use super::append::{
    append_vector_segment, expects_definition_only, fragment_reuse_affects_segments,
    index_field_path, is_definition_only_segment, merge_indices_with_unindexed_frags,
    partition_dormant_vector_segments, select_segments_to_merge, tagged_segment_coverage,
};
use super::frag_reuse::{OpenPurpose, SegmentRemappingPlan};
use super::loading::{DatasetIndexInternalExt, load_all_indices};
use super::scalar;
use super::segment::{
    index_is_usable, index_type_is_known, retain_committed_inverted_files,
    segment_has_inverted_details, segment_has_merge_primitive, segment_has_vector_details,
    segment_has_zonemap_details, unsupported_index_version, validate_segment_metadata,
};
use super::vector::details::{vector_index_details, vector_params_from_details};
use super::vector::ivf::{
    IVFIndex as LegacyIvfIndex, VectorSegmentCompatibility, optimize_vector_indices,
    select_steady_state_rebalance, vector_segment_compatibility,
};
use super::vector::{
    LogicalVectorIndex, VectorIndexParams, build_vector_model_segment, fresh_vector_segment_params,
};
use crate::dataset::Dataset;
use crate::dataset::transaction::{Operation, TransactionBuilder};
use crate::{Error, Result};

/// The indices an optimize pass may rewrite: each entry is one index name
/// with all of its segments, in name order.
///
/// An index this build cannot rewrite is skipped with a warning: an index type
/// it does not recognize, a segment newer than it reads, a covered index, an
/// externally maintained index, or a scalar index with nothing to do. A covered
/// or externally maintained index named in `index_names` is refused instead.
pub(crate) fn optimizable_indices<'a>(
    dataset: &Dataset,
    indices: &'a [IndexMetadata],
    options: &OptimizeOptions,
) -> Result<Vec<(String, Vec<&'a IndexMetadata>)>> {
    let indices_to_optimize = options
        .index_names
        .as_ref()
        .map(|names| names.iter().collect::<HashSet<_>>());
    let name_to_indices = indices
        .iter()
        .filter(|idx| {
            indices_to_optimize
                .as_ref()
                .is_none_or(|names| names.contains(&idx.name))
                && !is_system_index(idx)
        })
        .map(|idx| (idx.name.clone(), idx))
        .into_group_map();

    let mut optimizable = Vec::with_capacity(name_to_indices.len());
    for (name, deltas) in name_to_indices {
        if let Some(index) = deltas.iter().find(|idx| !index_type_is_known(idx)) {
            let type_url = index
                .index_details
                .as_ref()
                .map(|details| details.type_url.as_str())
                .unwrap_or("<legacy>");
            log::warn!(
                "Skipping optimization of index '{}' because this build does not recognize index type '{}'",
                index.name,
                type_url
            );
            continue;
        }

        if let Some(index) = deltas.iter().find(|index| {
            index.index_details.as_ref().is_some_and(|details| {
                scalar::SCALAR_INDEX_PLUGIN_REGISTRY
                    .get_plugin_by_details(details)
                    .is_ok_and(|plugin| !plugin.supports_optimization())
            })
        }) {
            if options.index_names.is_some() {
                return Err(Error::not_supported(format!(
                    "Index '{}' is maintained externally and does not support optimize_indices",
                    index.name
                )));
            }
            continue;
        }

        // Optimizing a covered index would republish its declaration on a
        // segment rebuilt without the carried values: `scan_vector_fragments`
        // projects the keyed field and `_rowid` only, and the scalar merges
        // reconstruct value plus row id.
        //
        // What decides is the caller's intent, not whether this group is
        // stale. An unfiltered `optimize_indices()` is a table-wide
        // maintenance request, and erroring aborts the loop before the
        // replacements accumulated for the other groups are committed -- so
        // one index this build cannot rebuild would leave every other index
        // on the table stale. Skip it with a warning instead.
        //
        // A caller that listed this index in `index_names` asked for it
        // specifically, so refuse out loud. The loop is already filtered by
        // that list, so reaching here with it set means this group was named.
        if let Some(covered) = deltas
            .iter()
            .find(|index| !index.covering_fields.is_empty())
        {
            if options.index_names.is_none() {
                log::warn!(
                    "Skipping index '{}': it declares covering fields {:?}, \
                     which no index builder writes or preserves yet.",
                    covered.name,
                    covered.covering_fields,
                );
                continue;
            }
            return Err(Error::index(format!(
                "Optimizing index '{}' is not supported: it declares \
                 covering fields {:?}, which no index builder writes or \
                 preserves yet",
                covered.name, covered.covering_fields,
            )));
        }

        // Optimizing a name means replacing its segments with one that
        // covers their union, which this build cannot compute when it
        // cannot read one of them: the merged segment would overlap the
        // segment left behind, and `Dataset::validate` calls that
        // corruption. Leave the whole name to a build that can read it.
        if let Some(max_supported_version) =
            deltas.iter().find_map(|idx| unsupported_index_version(idx))
        {
            log::warn!(
                "Index {} has a segment newer than version {}, which this build cannot read; \
                 skipping its optimization",
                name,
                max_supported_version,
            );
            continue;
        }
        if let Some(index) = deltas.iter().find(|index| !index_is_usable(index)) {
            log::warn!(
                "Index {} has a segment with unsupported field dependencies {:?}; \
                 skipping its optimization",
                name,
                index.dependency_fields(),
            );
            continue;
        }
        // Scalar indices have no rebalance concept, so skip them entirely
        // when every fragment is already covered and the caller hasn't
        // asked for retrain or an explicit delta merge. Vector indices
        // fall through and use a rebalance-aware no-op check inside
        // merge_indices_with_unindexed_frags.
        if !options.retrain
            && options.num_indices_to_merge.is_none_or(|n| n == 0)
            && index_group_is_scalar(dataset, &deltas)
            && index_group_has_no_unindexed(dataset, &deltas)
        {
            continue;
        }
        optimizable.push((name, deltas));
    }
    optimizable.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(optimizable)
}

fn index_group_is_scalar(dataset: &Dataset, deltas: &[&IndexMetadata]) -> bool {
    let Some(field_id) = deltas.first().and_then(|d| d.fields.first()) else {
        return false;
    };
    match dataset.schema().field_by_id(*field_id) {
        Some(field) => !is_vector_field(field.data_type()),
        None => false,
    }
}

fn index_group_has_no_unindexed(dataset: &Dataset, deltas: &[&IndexMetadata]) -> bool {
    let mut indexed = RoaringBitmap::new();
    for idx in deltas {
        if let Some(bitmap) = idx.fragment_bitmap.as_ref() {
            indexed |= bitmap;
        } else {
            // Pre-0.8 indices have no fragment bitmap; treat as needing optimize.
            return false;
        }
    }
    dataset
        .fragments()
        .iter()
        .all(|frag| indexed.contains(frag.id as u32))
}

fn is_vector_field(data_type: DataType) -> bool {
    match data_type {
        DataType::FixedSizeList(_, _) => true,
        DataType::List(inner) => {
            // A list of fixed-size lists is a multivector field.
            matches!(inner.data_type(), DataType::FixedSizeList(_, _))
        }
        _ => false,
    }
}

/// A plan to optimize the indices of a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeIndicesPlan {
    /// Identifies the plan. Every unit and result carries it, and a merge
    /// takes only the results of its own plan.
    pub plan_id: Uuid,
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
    pub plan_id: Uuid,
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
    /// Allow IVF partition splitting and joining within the task's merge window.
    /// Split builds keep this disabled so their results share the reference model.
    pub enable_partition_rebalancing: bool,
    /// Rebuild the index from a freshly trained model.
    pub retrain: bool,
}

/// Combine the results of one merge group into the segment to commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeIndexMerge {
    pub plan_id: Uuid,
    pub read_version: u64,
    /// The key task results of the same plan carry to reach this merge.
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
    pub plan_id: Uuid,
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

/// Next-fit bin packing: each item joins the last bin while it has room, or
/// opens a new one, so an item over the limit fills a bin alone and an item
/// never goes back to an earlier bin. Without a limit everything shares one
/// bin. Over items in fragment order every bin is a contiguous run of
/// fragments, which compaction needs to bin neighbors together.
fn pack_next_fit<T>(items: Vec<(T, u64)>, limit: Option<u64>) -> Vec<Vec<T>> {
    let limit = limit.unwrap_or(u64::MAX);
    // (items, rows used)
    let mut bins: Vec<(Vec<T>, u64)> = Vec::new();
    for (item, rows) in items {
        match bins.last_mut() {
            Some((bin, used)) if used.saturating_add(rows) <= limit => {
                bin.push(item);
                *used = used.saturating_add(rows);
            }
            _ => bins.push((vec![item], rows)),
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

fn validate_partition_rebalancing_options(options: &OptimizeOptions) -> Result<()> {
    if !options.enable_partition_rebalancing {
        return Ok(());
    }
    if options.num_indices_to_merge == Some(0) {
        return Err(Error::invalid_input(
            "enable_partition_rebalancing requires num_indices_to_merge to be positive or unset, got Some(0)",
        ));
    }
    if options.max_rows_per_task.is_some() || options.max_rows_per_segment.is_some() {
        return Err(Error::invalid_input(format!(
            "enable_partition_rebalancing cannot be combined with max_rows_per_task={:?} or max_rows_per_segment={:?}; partition rebalancing requires an unbounded merge task",
            options.max_rows_per_task, options.max_rows_per_segment,
        )));
    }
    Ok(())
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
            plan_id: Uuid::new_v4(),
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
    validate_partition_rebalancing_options(options)?;
    consolidate::validate_options(options)?;
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
        plan_id: Uuid::new_v4(),
        read_version: dataset.manifest.version,
        tasks: Vec::new(),
        merges: Vec::new(),
    };
    for (name, segments) in optimizable_indices(dataset, indices, options)? {
        let last = *segments.last().expect("an index has at least one segment");
        if consolidate::is_consolidation(options) {
            consolidate::plan_index(&mut plan, dataset, &segments, options, tagged).await?;
            continue;
        }
        let unindexed = dataset.unindexed_fragments(&name).await?;
        // Without an N:1 merge the single-process optimize rebuilds the
        // absorbed segments from scratch, and the only split that reproduces
        // it is none. Inverted stays unsplit under stable row ids until its
        // merge drops stale postings (issue #8085).
        let split = bounded
            && segment_has_merge_primitive(last)
            && !(segment_has_inverted_details(last) && dataset.manifest.uses_stable_row_ids());
        if !split {
            if bounded {
                log::warn!("Index '{name}' is not split, so it is optimized in one task");
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
    let (plan_id, read_version) = (plan.plan_id, plan.read_version);
    let group = plan.merges.len() as u32;
    let last = *segments.last().expect("an index has at least one segment");
    plan.tasks.push(OptimizeIndexTask {
        plan_id,
        read_version,
        group,
        fragments: unindexed
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect(),
        reference: last.clone(),
        old_segments: segments.iter().map(|segment| (*segment).clone()).collect(),
        num_indices_to_merge: options.num_indices_to_merge,
        enable_partition_rebalancing: options.enable_partition_rebalancing,
        retrain: options.retrain,
    });
    plan.merges.push(OptimizeIndexMerge {
        plan_id,
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
    let selected = segments[segments.len() - num_to_merge..].to_vec();
    let mut candidates = maintenance_candidates(dataset, name, selected)
        .await?
        .unwrap_or_default();
    retain_matching_versions(name, reference, &mut candidates);
    let absorbed: Vec<IndexMetadata> = candidates.into_iter().cloned().collect();
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
    let selected = select_segments_to_merge(dataset, segments, options);
    // The single-process merge removes a selected segment whose fragments are
    // all gone along with the rest; the split merge cannot absorb it.
    let dead: Vec<_> = selected
        .iter()
        .filter(|segment| {
            segment
                .effective_fragment_bitmap(&dataset.fragment_bitmap)
                .is_some_and(|bitmap| bitmap.is_empty())
        })
        .map(|segment| segment.uuid)
        .collect();
    let candidates = maintenance_candidates(dataset, name, selected).await?;
    let can_rebuild_zonemap = segment_has_zonemap_details(last) && candidates.is_some();
    let mut candidates = candidates.unwrap_or_default();
    let reference = if can_rebuild_zonemap {
        // Use the first live source's parameters for merging or rebuilding.
        let reference = candidates.first().copied().unwrap_or(last);
        if let Some(mut inputs) =
            plan_zonemap_rebuild(dataset, field_path, &candidates, &fragments, reference).await?
        {
            inputs.replaces.extend(dead);
            return Ok(Some(inputs));
        }
        reference
    } else {
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
        retain_matching_versions(name, last, &mut candidates);
        last
    };
    let absorbed = candidates.into_iter().cloned().collect();
    Ok(Some(SplitInputs {
        reference: reference.clone(),
        fragments,
        absorbed,
        replaces: dead,
    }))
}

/// Plan a fragment-local rebuild when the selected ZoneMap segments cannot be reused.
async fn plan_zonemap_rebuild(
    dataset: &Dataset,
    field_path: &str,
    selected: &[&IndexMetadata],
    fragments: &[Fragment],
    reference: &IndexMetadata,
) -> Result<Option<SplitInputs>> {
    // Metadata checks avoid loading all zone statistics during planning.
    if !scalar::zonemap::requires_rebuild_from_metadata(dataset, field_path, selected).await? {
        return Ok(None);
    }
    let mut coverage: RoaringBitmap = fragments
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    for segment in selected {
        coverage |= segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .unwrap_or_default();
    }
    Ok(Some(SplitInputs {
        reference: reference.clone(),
        fragments: dataset
            .fragments()
            .iter()
            .filter(|fragment| coverage.contains(fragment.id as u32))
            .cloned()
            .collect(),
        absorbed: Vec::new(),
        replaces: selected.iter().map(|segment| segment.uuid).collect(),
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
    let reference = train_model_segment(
        dataset,
        field_path,
        last,
        &params,
        None,
        options.progress.clone(),
    )
    .await?;
    Ok(Some(SplitInputs {
        reference,
        fragments: dataset.fragments().as_ref().clone(),
        absorbed: Vec::new(),
        replaces: segments.iter().map(|segment| segment.uuid).collect(),
    }))
}

/// Train a model for `segment`'s index, from `fragment_ids` or the whole
/// table, and describe it as a segment outside the manifest that build tasks
/// encode against.
async fn train_model_segment(
    dataset: &Dataset,
    field_path: &str,
    segment: &IndexMetadata,
    params: &VectorIndexParams,
    fragment_ids: Option<&[u32]>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<IndexMetadata> {
    let uuid = Uuid::new_v4();
    let frag_reuse_index = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
    let built = build_vector_model_segment(
        dataset,
        field_path,
        &segment.name,
        uuid,
        params,
        frag_reuse_index,
        fragment_ids,
        progress,
    )
    .await?;
    Ok(IndexMetadata {
        uuid,
        name: segment.name.clone(),
        fields: segment.fields.clone(),
        covering_fields: Vec::new(),
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(RoaringBitmap::new()),
        index_details: Some(Arc::new(vector_index_details(
            params,
            built.target_partition_size,
        ))),
        index_version: params.index_type().version(),
        created_at: Some(Utc::now()),
        base_id: None,
        files: Some(built.files),
        clustering_id: None,
        metadata: Default::default(),
    })
}

/// Select live sources for split maintenance. `None` defers all selected sources
/// because of a base or remap restriction; `Some` may contain no live sources.
async fn maintenance_candidates<'a>(
    dataset: &Dataset,
    name: &str,
    mut selected: Vec<&'a IndexMetadata>,
) -> Result<Option<Vec<&'a IndexMetadata>>> {
    selected.retain(|segment| {
        segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .is_some_and(|bitmap| !bitmap.is_empty())
    });
    let reason = if selected.iter().any(|segment| segment.base_id.is_some()) {
        Some("a segment lives in the base dataset of a shallow clone")
    } else if let Some(frag_reuse_index) =
        dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?
        && fragment_reuse_affects_segments(&frag_reuse_index, selected.iter().copied())
    {
        Some("a segment has a pending compaction remap")
    } else {
        None
    };
    if let Some(reason) = reason {
        log::warn!(
            "Index '{name}': appending new segments without merging existing ones because {reason}"
        );
        return Ok(None);
    }
    Ok(Some(selected))
}

/// Keep the default merge policy for indices that do not rebuild mixed versions.
fn retain_matching_versions(
    name: &str,
    reference: &IndexMetadata,
    candidates: &mut Vec<&IndexMetadata>,
) {
    if candidates
        .iter()
        .any(|segment| segment.index_version != reference.index_version)
    {
        log::warn!(
            "Index '{name}': appending new segments without merging existing ones because \
             their on-disk index versions differ"
        );
        candidates.clear();
    }
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

    let mut build_inputs: Vec<(u32, u64)> = fragments
        .iter()
        .map(|fragment| {
            let id = fragment.id as u32;
            (id, rows_by_fragment.get(&id).copied().unwrap_or(0))
        })
        .collect();
    build_inputs.sort_by_key(|(id, _)| *id);
    // A task's output is never split, so it holds no more than a segment may.
    let task_limit = [options.max_rows_per_task, options.max_rows_per_segment]
        .into_iter()
        .flatten()
        .min();
    let fragment_groups = pack_next_fit(build_inputs, task_limit);

    enum MergeInput {
        Absorbed(Box<IndexMetadata>),
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
        merge_inputs.push((MergeInput::Absorbed(Box::new(segment)), rows));
    }
    for group in fragment_groups {
        let rows = group
            .iter()
            .map(|id| rows_by_fragment.get(id).copied().unwrap_or(0))
            .sum();
        merge_inputs.push((MergeInput::Build(group), rows));
    }
    // In fragment order, so an output segment covers a contiguous run too.
    merge_inputs.sort_by_cached_key(|(input, _)| match input {
        MergeInput::Absorbed(segment) => segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .and_then(|bitmap| bitmap.min()),
        MergeInput::Build(fragments) => fragments.first().copied(),
    });

    for bin in pack_next_fit(merge_inputs, options.max_rows_per_segment) {
        let mut absorbed = Vec::new();
        let mut builds = Vec::new();
        for input in bin {
            match input {
                MergeInput::Absorbed(segment) => absorbed.push(*segment),
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
        let (plan_id, read_version) = (plan.plan_id, plan.read_version);
        let group = plan.merges.len() as u32;
        // A single task folds the old segments in itself, in one pass.
        let (absorbed_by_task, absorbed_by_merge) = if builds.len() == 1 {
            (absorbed, Vec::new())
        } else {
            (Vec::new(), absorbed)
        };
        for fragments in builds {
            plan.tasks.push(OptimizeIndexTask {
                plan_id,
                read_version,
                group,
                fragments,
                reference: reference.clone(),
                num_indices_to_merge: Some(absorbed_by_task.len()),
                enable_partition_rebalancing: false,
                old_segments: absorbed_by_task.clone(),
                retrain: false,
            });
        }
        plan.merges.push(OptimizeIndexMerge {
            plan_id,
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
        let mut options = OptimizeOptions::default().progress(progress);
        options.num_indices_to_merge = self.num_indices_to_merge;
        options.enable_partition_rebalancing = self.enable_partition_rebalancing;
        options.retrain = self.retrain;
        validate_partition_rebalancing_options(&options)?;
        let dataset = Arc::new(dataset_at_version(dataset, self.read_version).await?);
        let fragments = dataset
            .get_fragments_from_ids(&self.fragments)?
            .into_iter()
            .map(|fragment| fragment.metadata().clone())
            .collect::<Vec<_>>();
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
            plan_id: self.plan_id,
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
        for result in task_results {
            // `group` is unique within a plan only.
            if result.plan_id != self.plan_id {
                return Err(Error::invalid_input(format!(
                    "task result for group {} belongs to plan {}, not to this merge's plan {}",
                    result.group, result.plan_id, self.plan_id
                )));
            }
            if result.group != self.group {
                continue;
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
            plan_id: self.plan_id,
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
        index_details: last.index_details.clone().or_else(|| {
            Some(Arc::new(
                super::vector::details::vector_index_details_default(),
            ))
        }),
        index_version: last.index_version,
        created_at: Some(Utc::now()),
        base_id: None,
        files: Some(files),
        clustering_id: None,
        metadata: Default::default(),
    })
}

/// Commit the merged segments in one `CreateIndex` transaction.
///
/// The segments the results claim to replace are removed, exactly as the
/// single-process optimize removes what its merge reports. Fragments those
/// segments covered that no result covers (a rebuild missing results) go back
/// to unindexed, and the next optimize indexes them. Consolidation instead
/// requires complete surviving coverage for every replaced source. When called
/// on a refreshed dataset, it reuses staged output after concurrent append or
/// delete, but rejects sources that have been replaced since planning.
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
    let mut results = results;
    let read_version = if consolidate::is_consolidation(options) {
        consolidate::validate_commit(dataset, &planned, &mut results).await?;
        if results.is_empty() {
            return Ok(());
        }
        dataset.manifest.version
    } else {
        read_version
    };
    let committed_segments = if consolidate::is_consolidation(options) {
        load_all_indices(dataset).await?
    } else {
        load_all_indices(&planned).await?
    };

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

    use crate::index::frag_reuse_reader::tests::{
        fixture_with_index, install, persist_fixture, prepare,
    };
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_index::vector::{
        bq::RQBuildParams, hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, pq::PQBuildParams,
    };
    use lance_linalg::distance::MetricType;
    use lance_table::format::pb::fragment_reuse_index_details::InlineContent;
    use lance_testing::datagen::generate_random_array;
    use prost::Message;

    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::{UpdateBuilder, WriteParams};
    use crate::index::CreateIndexBuilder;

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
    fn test_pack_next_fit() {
        // Over the limit: a bin of its own. Otherwise the last bin while it
        // has room; "d" does not go back to "a", keeping bins contiguous.
        let bins = pack_next_fit(
            vec![("a", 300), ("b", 5000), ("c", 400), ("d", 200), ("e", 900)],
            Some(1000),
        );
        assert_eq!(bins, vec![vec!["a"], vec!["b"], vec!["c", "d"], vec!["e"]]);
        assert_eq!(
            pack_next_fit(vec![("a", 1), ("b", 2)], None),
            vec![vec!["a", "b"]]
        );
        assert!(pack_next_fit::<&str>(vec![], None).is_empty());
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
            assert_eq!(
                (result.plan_id, result.read_version),
                (plan.plan_id, plan.read_version)
            );
            built.push(result);
        }
        // A merge takes only the results of its own plan.
        let mut foreign = built[0].clone();
        foreign.plan_id = Uuid::new_v4();
        assert!(plan.merges[0].execute(&dataset, &[foreign]).await.is_err());
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

    #[rstest::rstest]
    #[tokio::test]
    async fn test_vector_retrain(#[values(false, true)] hnsw_rq: bool) {
        let test_dir = TempStrDir::default();
        let mut dataset = write_dataset(&test_dir, 1000, 500).await;
        if hnsw_rq {
            let (ivf, hnsw) = (IvfBuildParams::new(2), HnswBuildParams::default());
            let rq = RQBuildParams::default();
            let params = VectorIndexParams::with_ivf_hnsw_rq_params(MetricType::L2, ivf, hnsw, rq);
            let name = Some(VECTOR_INDEX.to_string());
            dataset
                .create_index(&["vector"], IndexType::Vector, name, &params, true)
                .await
                .unwrap();
        } else {
            create_vector_index(&mut dataset).await;
        }
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
        // The model segment carries the trained model only: no rows, no graphs.
        let opened =
            dataset.open_vector_index_from_metadata("vector", model, &NoOpMetricsCollector);
        assert_eq!(opened.await.unwrap().num_rows(), 0);
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

    #[rstest::rstest]
    #[case::merge_only(0, Some(4), None, 1)]
    #[case::single_build(4, Some(4), None, 1)]
    #[case::multiple_builds(8, Some(4), None, 1)]
    #[case::bounded_segments(8, Some(4), Some(8), 2)]
    #[case::unbounded(4, None, None, 1)]
    #[tokio::test]
    async fn test_split_zonemap_merges_without_rescanning(
        #[case] appended_rows: usize,
        #[case] task_limit: Option<u64>,
        #[case] segment_limit: Option<u64>,
        #[case] expected_segments: usize,
        #[values(false, true)] use_seeds: bool,
    ) {
        let dir = TempStrDir::default();
        let mut dataset = write_dataset(dir.as_str(), 8, 4).await;
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
            .with_params(&serde_json::json!({"rows_per_zone": 2, "use_seeds": use_seeds}));
        let mut staged = Vec::new();
        for fragment in dataset.get_fragments() {
            staged.push(
                CreateIndexBuilder::new(&mut dataset, &["id"], IndexType::ZoneMap, &params)
                    .name(SCALAR_INDEX.to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments(SCALAR_INDEX, "id", staged)
            .await
            .unwrap();
        let indexed_files = dataset
            .fragments()
            .iter()
            .flat_map(|fragment| fragment.files.iter().map(|file| file.path.clone()))
            .collect::<Vec<_>>();
        if appended_rows > 0 {
            append(&mut dataset, appended_rows, 4).await;
        }
        let options = OptimizeOptions::merge(2)
            .max_rows_per_task(task_limit)
            .max_rows_per_segment(segment_limit);
        dataset.object_store.as_ref().io_stats_incremental();
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        let plan: OptimizeIndicesPlan =
            serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
        assert_eq!(plan.tasks.len(), appended_rows / 4);
        assert_eq!(plan.merges.len(), expected_segments);
        let task_sources = plan
            .tasks
            .iter()
            .map(|task| task.old_segments.len())
            .sum::<usize>();
        let merge_sources = plan
            .merges
            .iter()
            .map(|merge| merge.old_segments.len())
            .sum::<usize>();
        if appended_rows == 4 {
            assert_eq!(task_sources, 2, "one build absorbs both existing segments");
            assert_eq!(merge_sources, 0);
        } else {
            assert_eq!(task_sources, 0);
            assert_eq!(merge_sources, 2);
        }
        let built =
            futures::future::try_join_all(plan.tasks.iter().map(|task| task.execute(&dataset)))
                .await
                .unwrap()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
        let built: Vec<IndexSegmentResult> =
            serde_json::from_str(&serde_json::to_string(&built).unwrap()).unwrap();
        let merged = futures::future::try_join_all(
            plan.merges
                .iter()
                .map(|merge| merge.execute(&dataset, &built)),
        )
        .await
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
        commit_optimize_indices(&mut dataset, merged, &options)
            .await
            .unwrap();
        let rescanned = dataset
            .object_store
            .as_ref()
            .io_stats_incremental()
            .requests
            .into_iter()
            .filter(|request| {
                indexed_files
                    .iter()
                    .any(|file| request.path.as_ref().ends_with(file))
            })
            .collect::<Vec<_>>();
        assert!(
            rescanned.is_empty(),
            "indexed data was rescanned: {rescanned:?}"
        );

        let dataset = Dataset::open(dir.as_str()).await.unwrap();
        let merged = segments(&dataset, SCALAR_INDEX).await;
        assert_eq!(merged.len(), expected_segments);
        assert_eq!(coverage(&merged), *dataset.fragment_bitmap);
        assert!(merged.iter().all(|segment| segment.index_version == 1));
        if let Some(limit) = segment_limit {
            assert!(
                merged
                    .iter()
                    .all(|segment| segment.fragment_bitmap.as_ref().unwrap().len() * 4 <= limit)
            );
        }
        assert_scalar_query_matches_scan(&dataset, "id >= 3").await;
    }

    async fn assert_scalar_query_matches_scan(dataset: &Dataset, predicate: &str) {
        let mut results = Vec::new();
        for use_index in [true, false] {
            let mut scan = dataset.scan();
            scan.project(&["id"])
                .unwrap()
                .filter(predicate)
                .unwrap()
                .use_scalar_index(use_index);
            if use_index {
                assert!(
                    scan.explain_plan(false)
                        .await
                        .unwrap()
                        .contains("ScalarIndexQuery")
                );
            }
            let batch = scan.try_into_batch().await.unwrap();
            let mut ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
            ids.sort_unstable();
            results.push(ids);
        }
        assert_eq!(results[0], results[1], "{predicate}");
    }

    #[rstest::rstest]
    #[case::physical_ids(false)]
    #[case::stable_ids(true)]
    #[tokio::test]
    async fn test_split_zonemap_updates_deletions_and_nulls(
        #[case] stable_ids: bool,
        #[values(1, 4)] task_limit: u64,
    ) {
        let batch = arrow_array::record_batch!(
            ("id", Int32, [0, 1, 2, 3, 4, 5, 6, 7]),
            (
                "value",
                Int32,
                [Some(0), None, None, None, Some(4), Some(5), None, Some(7)]
            )
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: stable_ids,
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
            .with_params(&serde_json::json!({"rows_per_zone": 2, "use_seeds": false}));
        let mut staged = Vec::new();
        for fragment in dataset.get_fragments() {
            staged.push(
                CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::ZoneMap, &params)
                    .name(SCALAR_INDEX.to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments(SCALAR_INDEX, "value", staged)
            .await
            .unwrap();
        dataset.delete("id = 0 OR id = 2 OR id = 3").await.unwrap();
        dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 4")
            .unwrap()
            .set("value", "NULL")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset
            .as_ref()
            .clone();
        dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")
            .unwrap()
            .set("value", "100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset
            .as_ref()
            .clone();
        let options = OptimizeOptions::merge(4).max_rows_per_task(Some(task_limit));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks.len(), if task_limit == 1 { 2 } else { 1 });
        if task_limit == 4 {
            assert!(plan.tasks[0].old_segments.len() >= 2);
        }
        let results = execute_plan(&dataset, &plan).await;
        commit_optimize_indices(&mut dataset, results, &options)
            .await
            .unwrap();
        assert_eq!(
            coverage(&segments(&dataset, SCALAR_INDEX).await),
            *dataset.fragment_bitmap
        );
        for predicate in [
            "value IS NULL",
            "value IS NOT NULL",
            "value = 4",
            "value = 100",
            "value >= 5",
        ] {
            assert_scalar_query_matches_scan(&dataset, predicate).await;
        }
    }

    #[rstest::rstest]
    #[case::btree(BuiltinIndexType::BTree, IndexType::BTree)]
    #[case::bitmap(BuiltinIndexType::Bitmap, IndexType::Bitmap)]
    #[case::zonemap(BuiltinIndexType::ZoneMap, IndexType::ZoneMap)]
    #[tokio::test]
    async fn test_split_scalar_preserves_shallow_clone_segments(
        #[case] builtin: BuiltinIndexType,
        #[case] index_type: IndexType,
    ) {
        let source_dir = TempStrDir::default();
        let clone_dir = TempStrDir::default();
        let mut source = write_dataset(source_dir.as_str(), 8, 4).await;
        source
            .create_index(
                &["id"],
                index_type,
                Some(SCALAR_INDEX.to_string()),
                &ScalarIndexParams::for_builtin(builtin),
                false,
            )
            .await
            .unwrap();
        let mut dataset = source
            .shallow_clone(clone_dir.as_str(), source.version_id(), None)
            .await
            .unwrap();
        let old = segments(&dataset, SCALAR_INDEX).await;
        assert!(old.iter().all(|segment| segment.base_id.is_some()));
        let options = OptimizeOptions::merge(1).max_rows_per_task(Some(4));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert!(plan.tasks.is_empty() && plan.merges.is_empty());

        append(&mut dataset, 4, 4).await;
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.tasks[0].fragments, vec![2]);
        assert!(plan.tasks[0].old_segments.is_empty());
        assert!(
            plan.merges
                .iter()
                .all(|merge| merge.old_segments.is_empty() && merge.replaces.is_empty())
        );
        let results = execute_plan(&dataset, &plan).await;
        commit_optimize_indices(&mut dataset, results, &options)
            .await
            .unwrap();
        let committed = segments(&dataset, SCALAR_INDEX).await;
        assert_eq!(committed.len(), old.len() + 1);
        assert!(old.iter().all(|segment| committed.contains(segment)));
        assert_eq!(coverage(&committed), *dataset.fragment_bitmap);
        assert_scalar_query_matches_scan(&dataset, "id >= 2 AND id < 10").await;

        // A retired base segment must not block maintenance of local live sources.
        let local = committed
            .iter()
            .find(|segment| segment.base_id.is_none())
            .unwrap()
            .uuid;
        dataset.delete("id < 8").await.unwrap();
        append(&mut dataset, 4, 4).await;
        let options = OptimizeOptions::merge(2).max_rows_per_task(Some(4));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.tasks[0].old_segments.len(), 1);
        assert_eq!(plan.tasks[0].old_segments[0].uuid, local);
        let results = execute_plan(&dataset, &plan).await;
        commit_optimize_indices(&mut dataset, results, &options)
            .await
            .unwrap();
        let committed = segments(&dataset, SCALAR_INDEX).await;
        assert_eq!(committed.len(), 1);
        assert!(committed[0].base_id.is_none());
        assert_eq!(coverage(&committed), *dataset.fragment_bitmap);
        assert_scalar_query_matches_scan(&dataset, "id >= 2 AND id < 10").await;
    }

    #[rstest::rstest]
    #[case::split(Some(4), None)]
    #[case::bounded_segments(Some(4), Some(4))]
    #[case::unsplit(None, None)]
    #[tokio::test]
    async fn test_zonemap_preserves_shared_remap_policy(
        #[case] task_limit: Option<u64>,
        #[case] segment_limit: Option<u64>,
        #[values(1, 2)] source_count: usize,
    ) {
        let batch = arrow_array::record_batch!(
            ("id", Int32, [0, 1, 2, 3, 4, 5, 6, 7]),
            (
                "value",
                Float64,
                [
                    Some(0.0),
                    Some(f64::NAN),
                    None,
                    Some(3.0),
                    Some(4.0),
                    Some(5.0),
                    None,
                    Some(7.0)
                ]
            )
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
            .with_params(&serde_json::json!({"rows_per_zone": 4, "use_seeds": false}));
        let mut staged = Vec::new();
        let groups = if source_count == 1 {
            vec![vec![0, 1]]
        } else {
            vec![vec![0], vec![1]]
        };
        for fragments in groups {
            staged.push(
                CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::ZoneMap, &params)
                    .name(SCALAR_INDEX.to_string())
                    .fragments(fragments)
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments(SCALAR_INDEX, "value", staged)
            .await
            .unwrap();
        dataset.delete("id = 0").await.unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 4,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let old = segments(&dataset, SCALAR_INDEX).await;
        let error = dataset
            .merge_existing_index_segments(old.clone())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(
            error.to_string().contains("counts are no longer exact"),
            "{error}"
        );

        let options = OptimizeOptions::merge(2)
            .max_rows_per_task(task_limit)
            .max_rows_per_segment(segment_limit);
        if task_limit.is_some() {
            let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
            assert!(plan.tasks.is_empty() && plan.merges.is_empty());
        }
        let appended = arrow_array::record_batch!(
            ("id", Int32, [8, 9]),
            ("value", Float64, [Some(8.0), None])
        )
        .unwrap();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
                None,
            )
            .await
            .unwrap();
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks.len(), 1);
        if task_limit.is_some() {
            assert!(plan.tasks[0].old_segments.is_empty());
            assert_eq!(
                plan.tasks[0].fragments,
                vec![dataset.fragments().last().unwrap().id as u32]
            );
            assert!(
                plan.merges
                    .iter()
                    .all(|merge| merge.old_segments.is_empty() && merge.replaces.is_empty())
            );
        } else {
            assert_eq!(plan.tasks[0].old_segments.len(), source_count);
        }
        let results = execute_plan(&dataset, &plan).await;
        commit_optimize_indices(&mut dataset, results, &options)
            .await
            .unwrap();
        let merged = segments(&dataset, SCALAR_INDEX).await;
        if task_limit.is_some() {
            assert_eq!(merged.len(), source_count + 1);
            assert!(
                old.iter()
                    .all(|source| merged.iter().any(|segment| segment.uuid == source.uuid))
            );
        } else {
            assert_eq!(merged.len(), 1);
            assert!(old.iter().all(|source| merged[0].uuid != source.uuid));
        }
        assert!(merged.iter().all(|segment| segment.index_version == 1));
        assert_eq!(coverage(&merged), *dataset.fragment_bitmap);
        for predicate in ["value IS NULL", "value >= 3.0", "value = 8.0"] {
            assert_scalar_query_matches_scan(&dataset, predicate).await;
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_zonemap_preserves_shared_tagged_policy(
        #[values(false, true)] multiple_sources: bool,
        #[values(false, true)] append_data: bool,
    ) {
        let mut dataset = fixture_with_index(IndexType::ZoneMap).await;
        if multiple_sources {
            let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
            let mut staged = Vec::new();
            for fragment in dataset.get_fragments() {
                staged.push(
                    CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::ZoneMap, &params)
                        .name("i_idx".into())
                        .replace(true)
                        .fragments(vec![fragment.id() as u32])
                        .execute_uncommitted()
                        .await
                        .unwrap(),
                );
            }
            dataset
                .commit_existing_index_segments("i_idx", "i", staged)
                .await
                .unwrap();
        }
        dataset.delete("i = 3").await.unwrap();
        let (transition, destinations) = prepare(&dataset).await;
        install(
            &mut dataset,
            InlineContent {
                legacy_versions: vec![],
                transitions: vec![transition],
            }
            .encode_to_vec(),
            destinations,
            false,
        )
        .await;
        let stored = load_all_indices(&dataset).await.unwrap().as_ref().clone();
        persist_fixture(&mut dataset, stored).await;
        if append_data {
            let batch = arrow_array::record_batch!(("i", Int32, [8, 9])).unwrap();
            dataset
                .append(
                    RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                    None,
                )
                .await
                .unwrap();
        }
        let options = OptimizeOptions::merge(2)
            .max_rows_per_task(Some(4))
            .max_rows_per_segment(Some(4));
        let before = load_all_indices(&dataset).await.unwrap();
        let read_version = dataset.version_id();
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert!(plan.tasks.is_empty() && plan.merges.is_empty());
        dataset.optimize_indices(&options).await.unwrap();
        assert_eq!(dataset.version_id(), read_version);
        assert_eq!(*load_all_indices(&dataset).await.unwrap(), *before);
        let mut named = options.clone();
        named.index_names = Some(vec!["i_idx".into()]);
        let error = plan_optimize_indices(&dataset, &named).await.unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(
            error.to_string().contains("tagged fragment reuse history"),
            "{error}"
        );
        for value in 0..if append_data { 10 } else { 8 } {
            let predicate = format!("i = {value}");
            let mut actual = Vec::new();
            for use_index in [true, false] {
                let mut scan = dataset.scan();
                scan.filter(&predicate).unwrap().use_scalar_index(use_index);
                if use_index {
                    assert!(
                        scan.explain_plan(false)
                            .await
                            .unwrap()
                            .contains("ScalarIndexQuery")
                    );
                }
                actual.push(scan.try_into_batch().await.unwrap());
            }
            assert_eq!(actual[0], actual[1], "{predicate}");
            assert_eq!(actual[0].num_rows(), usize::from(value != 3));
        }
    }

    #[rstest::rstest]
    #[case::zone_size_full(4, true, false)]
    #[case::zone_size_partial(4, true, true)]
    #[case::null_policy_full(2, false, false)]
    #[case::null_policy_partial(2, false, true)]
    #[tokio::test]
    async fn test_zonemap_distributed_rebuild_incompatible_parameters(
        #[case] second_zone_size: u64,
        #[case] second_null_policy: bool,
        #[case] partial_commit: bool,
    ) {
        let mut dataset = write_dataset("memory://", 8, 4).await;
        let mut staged = Vec::new();
        for (fragment, rows_per_zone, store_null_bitmap) in
            [(0, 2, true), (1, second_zone_size, second_null_policy)]
        {
            let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
                .with_params(&serde_json::json!({"rows_per_zone": rows_per_zone, "store_null_bitmap": store_null_bitmap}));
            staged.push(
                CreateIndexBuilder::new(&mut dataset, &["id"], IndexType::ZoneMap, &params)
                    .name(SCALAR_INDEX.into())
                    .fragments(vec![fragment])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments(SCALAR_INDEX, "id", staged)
            .await
            .unwrap();
        append(&mut dataset, 4, 4).await;
        let options = OptimizeOptions::merge(2)
            .max_rows_per_task(Some(4))
            .max_rows_per_segment(Some(4));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks.len(), 3);
        assert!(plan.tasks.iter().all(|task| task.old_segments.is_empty()));
        let results = execute_plan(&dataset, &plan).await;
        if partial_commit {
            let old = segments(&dataset, SCALAR_INDEX).await;
            commit_optimize_indices(
                &mut dataset,
                results[1..].to_vec(),
                &OptimizeOptions::default(),
            )
            .await
            .unwrap();
            let committed = segments(&dataset, SCALAR_INDEX).await;
            assert_eq!(coverage(&committed), RoaringBitmap::from_iter([1, 2]));
            assert!(committed.iter().all(|segment| !old.contains(segment)));
            assert_eq!(
                dataset
                    .unindexed_fragments(SCALAR_INDEX)
                    .await
                    .unwrap()
                    .iter()
                    .map(|fragment| fragment.id)
                    .collect::<Vec<_>>(),
                vec![0],
            );
            assert_scalar_query_matches_scan(&dataset, "id >= 2 AND id < 10").await;

            let retry = plan_optimize_indices(&dataset, &options).await.unwrap();
            assert_eq!(retry.tasks.len(), 1);
            assert_eq!(retry.tasks[0].fragments, vec![0]);
            let results = execute_plan(&dataset, &retry).await;
            commit_optimize_indices(&mut dataset, results, &OptimizeOptions::default())
                .await
                .unwrap();
        } else {
            commit_optimize_indices(&mut dataset, results, &OptimizeOptions::default())
                .await
                .unwrap();
        }
        assert_eq!(
            coverage(&segments(&dataset, SCALAR_INDEX).await),
            *dataset.fragment_bitmap
        );
        assert_scalar_query_matches_scan(&dataset, "id >= 2 AND id < 10").await;
    }
}
