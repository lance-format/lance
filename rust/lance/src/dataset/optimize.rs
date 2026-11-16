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
use std::sync::Arc;

use lance_core::memory::{ObservedOperation, OperationScope};
use lance_core::utils::tracing::{DATASET_COMPACTING_EVENT, TRACE_DATASET_EVENTS};
use lance_table_runtime::optimize as runtime;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::index::DatasetIndexRemapperOptions;
use crate::{Dataset, Result};

mod hooks;
mod planner;
pub mod remapping;

pub use planner::{
    CompactionPlan, CompactionPlanner, CompactionPlanningResult, DefaultCompactionPlanner,
    plan_compaction,
};
pub use remapping::{IgnoreRemap, IndexRemapper, IndexRemapperOptions, RemappedIndex};
pub use runtime::{
    COMPACTION_CONFIG_PREFIX, CompactionGroup, CompactionMetrics, CompactionMode,
    CompactionOptions, CompactionScope, CompactionTaskKind, IndexRemapMode, RepackedFiles,
    RewriteResult, SourceBudgetLimit, SourceBudgetMode, SourceBudgetSummary, TaskData,
};

use hooks::LanceCompactionHooks;

/// Compacts the files in the dataset without reordering them.
///
/// By default, this does a few things:
///  * Removes deleted rows from fragments.
///  * Removes dropped columns from fragments.
///  * Merges fragments that are too small.
///
/// With [`CompactionOptions::max_data_files_per_fragment`] set, it also
/// repacks the columns of a fragment spread over too many data files into
/// fewer files without moving its rows. A run that both
/// rewrites fragments and repacks columns commits the two separately (see
/// [`commit_compaction`]). [`CompactionOptions::scope`] limits a run to one
/// kind of task.
///
/// This method tries to preserve the insertion order of rows in the dataset.
///
/// If no compaction is needed, this method will not make a new version of the table.
pub async fn compact_files(
    dataset: &mut Dataset,
    options: CompactionOptions,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>, // These will be deprecated later
) -> Result<CompactionMetrics> {
    info!(target: TRACE_DATASET_EVENTS, event=DATASET_COMPACTING_EVENT, uri = &dataset.table_state.uri);
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
    let scope = OperationScope::start(ObservedOperation::Compaction, || None)?;
    let plan = planner.plan(dataset).await?;
    let bindings = LanceCompactionHooks::new(dataset, Some(remap_options.as_ref()));
    let mut table = dataset.table_view().to_table();
    let result = runtime::execute_compaction_plan_with_stages(
        &mut table,
        plan.into(),
        &bindings.runtime(),
        &scope.stages(),
    )
    .await;
    (dataset.table_state, dataset.manifest, dataset.refs) = table.into_parts();
    let metrics = result?;
    scope.finish();
    Ok(metrics)
}

/// Commit the results of file compaction.
///
/// It is not required that all tasks are passed to this method. If some failed,
/// they can be omitted and the successful tasks can be committed. However, once
/// some of the tasks have been committed, the remainder of the tasks will not
/// be able to be committed and should be considered cancelled.
///
/// Fragment rewrites commit as one `Operation::Rewrite`, then column repacks
/// as one `Operation::DataReplacement` that moves values without changing
/// them. A fragment in both kinds of result is rejected before either commit.
/// If the repack commit fails, the rewrite stays committed and `dataset` is
/// left at its version; plan again rather than passing the same results back,
/// since their rewrites would conflict with the commit that landed.
///
/// With stable row IDs, tasks that recorded their row-address moves (see
/// [`CompactionOptions::defer_index_remap`]) commit them even if `options` does not
/// set `defer_index_remap`; either all rewrite results or none must have
/// recorded them. Results that did not record commit without recording, also
/// on a table that has a fragment reuse index.
pub async fn commit_compaction(
    dataset: &mut Dataset,
    completed_tasks: Vec<RewriteResult>,
    remap_options: Arc<dyn IndexRemapperOptions>,
    options: &CompactionOptions,
) -> Result<CompactionMetrics> {
    let mut table = dataset.table_view().to_table();
    let bindings = LanceCompactionHooks::new(dataset, Some(remap_options.as_ref()));
    let result = runtime::commit_compaction_with_hooks(
        &mut table,
        completed_tasks,
        options,
        &bindings.runtime(),
    )
    .await;
    // If a later repack fails, the earlier rewrite can already be published.
    (dataset.table_state, dataset.manifest, dataset.refs) = table.into_parts();
    result
}

/// A standalone task that can be serialized and sent to another machine for
/// execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionTask {
    pub task: TaskData,
    pub read_version: u64,
    pub options: CompactionOptions,
}

impl CompactionTask {
    /// Run the compaction task and return the result. A
    /// [`CompactionTaskKind::RewriteFragments`] task rewrites its fragments
    /// into new ones; a [`CompactionTaskKind::RepackColumns`] task writes new
    /// data files for its fragment.
    ///
    /// This result should be later passed to
    /// [`crate::dataset::optimize::commit_compaction`] to commit
    /// the changes to the dataset.
    ///
    /// Note: you should pass the version of the dataset that is the same as
    /// the read version for this task (the same version from which the
    /// compaction was planned).
    pub async fn execute(&self, dataset: &Dataset) -> Result<RewriteResult> {
        let table = dataset.table_view().to_table();
        let bindings = LanceCompactionHooks::new(dataset, None);
        runtime::execution::execute_task(
            &table,
            self.read_version,
            &self.task,
            &self.options,
            &bindings.runtime(),
        )
        .await
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
use tests::rewrite_files;
