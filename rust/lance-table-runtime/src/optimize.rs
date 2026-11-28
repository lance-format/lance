// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan, execute and publish table compaction.
//!
//! Use [compact_files] for a local run. To distribute work, call
//! [plan_compaction], serialize the [CompactionTask]s returned by
//! [CompactionPlan::compaction_tasks], then execute them against the same
//! table and pass the results to [commit_compaction].

use futures::{StreamExt, TryStreamExt};
use lance_core::memory::{ObservedOperation, OperationScope, StageReporter};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{DATASET_COMPACTING_EVENT, TRACE_DATASET_EVENTS};
use lance_core::{Error, Result};
use tracing::info;

use crate::Table;

mod binary_copy;
mod column_layout;
#[doc(hidden)]
pub mod commit;
#[doc(hidden)]
pub mod execution;
mod hooks;
#[doc(hidden)]
pub mod planning;
mod reader;
pub mod remapping;
#[doc(hidden)]
pub mod repack;

pub use column_layout::FragmentColumnLayoutStats;
pub use commit::{commit_compaction, commit_compaction_with_hooks};
pub use execution::{CompactionMetrics, CompactionTask, RepackedFiles, RewriteResult};
pub use hooks::{
    CompactionHooks, CompactionIndexHook, CompactionIndexSegment, IndexRemapperFactory,
};
pub use planning::{
    COMPACTION_CONFIG_PREFIX, CompactionGroup, CompactionMode, CompactionOptions, CompactionPlan,
    CompactionPlanner, CompactionPlanningResult, CompactionScope, CompactionTaskKind,
    DefaultCompactionPlanner, IndexRemapMode, SourceBudgetLimit, SourceBudgetMode,
    SourceBudgetSummary, TaskData, plan_compaction,
};
pub use remapping::{IgnoreRemap, IndexRemapper, RemapResult, RemappedIndex};

use execution::refuse_stale_output_clustering;
use planning::validate_fragment_groups_preserve_index_boundaries;

/// Merge small fragments and materialize deletions and overlays without
/// reordering the rows within a task. See [CompactionOptions] for repacking
/// columns without moving rows.
pub async fn compact_files(
    table: &mut Table,
    options: CompactionOptions,
) -> Result<CompactionMetrics> {
    info!(target: TRACE_DATASET_EVENTS, event=DATASET_COMPACTING_EVENT, uri = table.uri());
    let planner = DefaultCompactionPlanner::new(options)?;
    Box::pin(compact_files_with_planner(table, &planner)).await
}

/// Compact a table using a caller-supplied planning strategy.
pub async fn compact_files_with_planner(
    table: &mut Table,
    planner: &dyn CompactionPlanner,
) -> Result<CompactionMetrics> {
    let scope = OperationScope::start(ObservedOperation::Compaction, || None)?;
    let plan = planner.plan(table).await?;
    let metrics = execute_compaction_plan_with_stages(
        table,
        plan,
        &CompactionHooks::default(),
        &scope.stages(),
    )
    .await?;
    scope.finish();
    Ok(metrics)
}

/// Execute and publish an already computed plan using locally bound hooks.
pub async fn execute_compaction_plan(
    table: &mut Table,
    plan: CompactionPlan,
    hooks: &CompactionHooks<'_>,
) -> Result<CompactionMetrics> {
    let scope = OperationScope::start(ObservedOperation::Compaction, || None)?;
    let metrics = execute_compaction_plan_with_stages(table, plan, hooks, &scope.stages()).await?;
    scope.finish();
    Ok(metrics)
}

/// Execute a plan within the caller's compaction operation.
#[doc(hidden)]
pub async fn execute_compaction_plan_with_stages(
    table: &mut Table,
    plan: CompactionPlan,
    hooks: &CompactionHooks<'_>,
    stages: &StageReporter,
) -> Result<CompactionMetrics> {
    let options = &plan.options;
    if !hooks
        .remapper
        .is_some_and(|remapper| remapper.supports_vector_segment_replacement())
        || options.defer_index_remap
        || table.manifest.uses_stable_row_ids()
    {
        validate_fragment_groups_preserve_index_boundaries(
            table,
            plan.tasks.iter().map(|task| task.fragments.as_slice()),
            hooks,
        )
        .await?;
    }

    // A custom plan can bypass the option validation normally done at planning.
    if options.num_threads == Some(0) {
        return Err(Error::invalid_input(
            "CompactionOptions::num_threads must be greater than 0 (use None for the default)",
        ));
    }
    if plan.tasks.is_empty() {
        return Ok(CompactionMetrics::default());
    }
    let has_rewrites = plan
        .tasks
        .iter()
        .any(|task| task.kind == CompactionTaskKind::RewriteFragments);
    if has_rewrites && !options.defer_index_remap {
        commit::reject_eager_remap_on_tagged_fri(table).await?;
    }
    for task in &plan.tasks {
        refuse_stale_output_clustering(&table.manifest, task)?;
    }

    let concurrency = options
        .num_threads
        .unwrap_or_else(get_num_compute_intensive_cpus);
    let snapshot = &*table;
    let read_version = plan.read_version;
    // Concurrent tasks share a stage and still execute on the planned snapshot.
    let execution_stage = stages.start(if has_rewrites {
        "rewrite_files"
    } else {
        "repack_files"
    });
    let results = futures::stream::iter(plan.tasks)
        .map(move |task| async move {
            execution::execute_task_inner(snapshot, read_version, &task, options, hooks).await
        })
        .buffer_unordered(concurrency)
        .try_collect()
        .await?;
    execution_stage.complete();
    commit::commit_compaction_inner(table, results, options, hooks, stages).await
}
