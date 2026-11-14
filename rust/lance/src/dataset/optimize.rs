// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Table maintenance for optimizing table layout.
//!
//! As a table is updated, its layout can become suboptimal. For example, if
//! a series of small streaming appends are performed, eventually there will be
//! a large number of small files. This imposes an overhead to track the large
//! number of files and for very small files can make it harder to read data
//! efficiently. In this case, files can be compacted into fewer larger files.
//!
//! To compact files in a table, use the [compact_files] method. This currently
//! can compact in two cases:
//!
//! 1. If a fragment has fewer rows than the target number of rows per fragment.
//!    A reliable main data file must also be below the configured small-file
//!    size threshold. The fragment must have neighbors that are candidates for
//!    compaction.
//! 2. If a fragment has a higher percentage of deleted rows than the provided
//!    threshold.
//!
//! In addition to the rules above there may be restrictions due to indexes.
//! When a fragment is compacted its row ids change and any index that contained
//! that fragment will be remapped.  However, we cannot combine indexed fragments
//! with unindexed fragments.
//!
//! ```rust
//! # use std::sync::Arc;
//! # use tokio::runtime::Runtime;
//! # use arrow_array::{RecordBatch, RecordBatchIterator, Int64Array};
//! # use arrow_schema::{Schema, Field, DataType};
//! use lance::{dataset::WriteParams, Dataset, dataset::optimize::compact_files};
//! // Remapping indices is ignored in this example.
//! use lance::dataset::optimize::IgnoreRemap;
//!
//! # let mut rt = Runtime::new().unwrap();
//! # rt.block_on(async {
//! #
//! # let test_dir = lance_core::utils::tempfile::TempStrDir::default();
//! # let uri = test_dir.to_string();
//! let schema = Arc::new(Schema::new(vec![Field::new("test", DataType::Int64, false)]));
//! let data = RecordBatch::try_new(
//!     schema.clone(),
//!     vec![Arc::new(Int64Array::from_iter_values(0..10_000))]
//! ).unwrap();
//! let reader = RecordBatchIterator::new(vec![Ok(data)], schema);
//!
//! // Write 100 small files
//! let write_params = WriteParams { max_rows_per_file: 100, ..Default::default()};
//! let mut dataset = Dataset::write(reader, &uri, Some(write_params)).await.unwrap();
//! assert_eq!(dataset.get_fragments().len(), 100);
//!
//! // Use compact_files() to consolidate the data to 1 fragment
//! let metrics = compact_files(&mut dataset, Default::default(), None).await.unwrap();
//! assert_eq!(metrics.fragments_removed, 100);
//! assert_eq!(metrics.fragments_added, 1);
//! assert_eq!(dataset.get_fragments().len(), 1);
//! # })
//! ```
//!
//! ## Distributed execution
//!
//! The [compact_files] method internally can use multiple threads, but
//! sometimes you might want to run it across multiple machines. To do this,
//! use the task API.
//!
//! ```text
//!                                      ┌──► CompactionTask.execute() ─► RewriteResult ─┐
//! plan_compaction() ─► CompactionPlan ─┼──► CompactionTask.execute() ─► RewriteResult ─┼─► commit_compaction()
//!                                      └──► CompactionTask.execute() ─► RewriteResult ─┘
//! ```
//!
//! [plan_compaction()] produces a [CompactionPlan]. This can be split into multiple
//! [CompactionTask], which can be serialized and sent to other machines. Calling
//! [CompactionTask::execute()] performs the compaction and returns a [RewriteResult].
//! The [RewriteResult] can be sent back to the coordinator, which can then call
//! [commit_compaction()] to commit the changes to the dataset.
//!
//! It's not required that all tasks are passed to [commit_compaction]. If some
//! didn't complete successfully or before a deadline, they can be omitted and
//! the successful tasks can be committed. You can also commit in batches if
//! you wish. As long as the tasks don't rewrite any of the same fragments,
//! they can be committed in any order.
use std::borrow::Cow;
use std::sync::Arc;

use super::index::DatasetIndexRemapperOptions;
use crate::Dataset;
use crate::Result;
use futures::{StreamExt, TryStreamExt};
use lance_core::Error;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{DATASET_COMPACTING_EVENT, TRACE_DATASET_EVENTS};
use tracing::info;

pub(super) mod binary_copy;
pub(crate) mod commit;
pub(crate) mod execution;
mod planning;
pub mod remapping;

pub use commit::commit_compaction;
pub use execution::{CompactionMetrics, CompactionTask, RewriteResult};
use execution::{refuse_stale_output_clustering, rewrite_files};
use planning::validate_fragment_groups_preserve_index_boundaries;
pub use planning::{
    COMPACTION_CONFIG_PREFIX, CompactionGroup, CompactionMode, CompactionOptions, CompactionPlan,
    CompactionPlanner, CompactionPlanningResult, DefaultCompactionPlanner, IndexRemapMode,
    SourceBudgetLimit, SourceBudgetMode, SourceBudgetSummary, TaskData, plan_compaction,
};
pub use remapping::{IgnoreRemap, IndexRemapper, IndexRemapperOptions, RemappedIndex};

/// Compacts the files in the dataset without reordering them.
///
/// By default, this does a few things:
///  * Removes deleted rows from fragments.
///  * Removes dropped columns from fragments.
///  * Merges fragments that are too small.
///
/// This method tries to preserve the insertion order of rows in the dataset.
///
/// If no compaction is needed, this method will not make a new version of the table.
pub async fn compact_files(
    dataset: &mut Dataset,
    options: CompactionOptions,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>, // These will be deprecated later
) -> Result<CompactionMetrics> {
    info!(target: TRACE_DATASET_EVENTS, event=DATASET_COMPACTING_EVENT, uri = &dataset.uri);
    let remap_options =
        remap_options.unwrap_or_else(|| Arc::new(DatasetIndexRemapperOptions::default()));
    let planner =
        DefaultCompactionPlanner::new(options)?.with_index_remapper(remap_options.as_ref());
    // Keep the compaction state machine from inflating callers' futures.
    Box::pin(compact_files_with_planner(
        dataset,
        Some(remap_options),
        &planner,
    ))
    .await
}

pub async fn compact_files_with_planner(
    dataset: &mut Dataset,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>, // These will be deprecated later
    planner: &dyn CompactionPlanner,
) -> Result<CompactionMetrics> {
    let remap_options =
        remap_options.unwrap_or_else(|| Arc::new(DatasetIndexRemapperOptions::default()));
    let compaction_plan: CompactionPlan = planner.plan(dataset).await?;
    if !remap_options.supports_vector_segment_replacement()
        || compaction_plan.options.defer_index_remap
        || dataset.manifest.uses_stable_row_ids()
    {
        validate_fragment_groups_preserve_index_boundaries(
            dataset,
            compaction_plan
                .tasks
                .iter()
                .map(|task| task.fragments.as_slice()),
        )
        .await?;
    }

    // `buffer_unordered(0)` below never pulls a task from the stream and
    // returns `Pending` with nothing left to wake it, so zero threads would
    // leave compaction waiting forever. Checked against the plan so plans from
    // a custom planner are covered too.
    if compaction_plan.options.num_threads == Some(0) {
        return Err(Error::invalid_input(
            "CompactionOptions::num_threads must be greater than 0 (use None for the default)",
        ));
    }

    // A tagged FRI history is maintained by appending transitions to the
    // tagged entry, which only the deferred-remap commit path does; eager
    // remapping would rewrite provenance the tagged reader depends on.
    // Checked before any file is rewritten (commit_compaction re-checks for
    // callers that commit externally planned results).
    if !compaction_plan.options.defer_index_remap
        && dataset.manifest.writer_feature_flags
            & lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX
            != 0
        && crate::index::loading::load_all_indices(dataset)
            .await?
            .iter()
            .any(lance_table::system_index::frag_reuse::metadata::is_tagged)
    {
        return Err(Error::not_supported(
            "Compacting a table with a tagged fragment reuse history requires \
             defer_index_remap; set defer_index_remap, or upgrade to a writer that \
             remaps tagged histories eagerly",
        ));
    }

    // If nothing to compact, don't make a commit.
    if compaction_plan.tasks().is_empty() {
        return Ok(CompactionMetrics::default());
    }
    for task in compaction_plan.tasks() {
        refuse_stale_output_clustering(&dataset.manifest, task)?;
    }

    let dataset_ref = &dataset.clone();

    let result_stream = futures::stream::iter(compaction_plan.tasks)
        .map(|task| rewrite_files(Cow::Borrowed(dataset_ref), task, &compaction_plan.options))
        .buffer_unordered(
            compaction_plan
                .options
                .num_threads
                .unwrap_or_else(get_num_compute_intensive_cpus),
        );

    let completed_tasks: Vec<RewriteResult> = result_stream.try_collect().await?;
    let metrics = commit_compaction(
        dataset,
        completed_tasks,
        remap_options,
        &compaction_plan.options,
    )
    .await?;

    Ok(metrics)
}

#[cfg(test)]
mod tests;
