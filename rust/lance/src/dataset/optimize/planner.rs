// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use lance_table_runtime::optimize as runtime;
use serde::{Deserialize, Serialize};

use crate::{Dataset, Result};

use super::hooks::LanceCompactionHooks;
use super::remapping::IndexRemapperOptions;
use super::{CompactionGroup, CompactionOptions, CompactionTask, SourceBudgetSummary, TaskData};

/// A plan for compacting the files of a dataset.
///
/// Use [plan_compaction] to build a plan, then call [Self::compaction_tasks]
/// to execute its tasks locally or on workers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompactionPlan {
    pub tasks: Vec<TaskData>,
    pub read_version: u64,
    pub options: CompactionOptions,
}

impl From<runtime::CompactionPlan> for CompactionPlan {
    fn from(plan: runtime::CompactionPlan) -> Self {
        Self {
            tasks: plan.tasks,
            read_version: plan.read_version,
            options: plan.options,
        }
    }
}

impl From<CompactionPlan> for runtime::CompactionPlan {
    fn from(plan: CompactionPlan) -> Self {
        Self {
            tasks: plan.tasks,
            read_version: plan.read_version,
            options: plan.options,
        }
    }
}

impl CompactionPlan {
    /// Retrieve standalone tasks that can be executed in a distributed fashion.
    pub fn compaction_tasks(&self) -> impl Iterator<Item = CompactionTask> + '_ {
        let read_version = self.read_version;
        let options = self.options.clone();
        self.tasks.iter().map(move |task| CompactionTask {
            task: task.clone(),
            read_version,
            options: options.clone(),
        })
    }

    /// The number of tasks in the plan.
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// The version of the dataset that was read to produce this plan.
    pub fn read_version(&self) -> u64 {
        self.read_version
    }

    /// The options used to produce this plan.
    pub fn options(&self) -> &CompactionOptions {
        &self.options
    }

    #[cfg(test)]
    pub(super) fn tasks(&self) -> &[TaskData] {
        &self.tasks
    }
}

/// A compaction plan together with its source-budget admission outcome.
///
/// Obtained from [`DefaultCompactionPlanner::plan_with_budget`]. The ordinary
/// [`CompactionPlan`] remains unchanged for existing callers and serialized plans.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompactionPlanningResult {
    /// The tasks admitted under the configured source-budget mode.
    pub plan: CompactionPlan,
    /// Admitted work and the first task that could not fit.
    pub source_budget: SourceBudgetSummary,
}

/// Trait for implementing custom compaction planning strategies.
///
/// This trait allows users to define their own compaction strategies by implementing
/// the `plan` method. The default implementation is provided by [`DefaultCompactionPlanner`].
#[async_trait]
pub trait CompactionPlanner: Send + Sync {
    /// Build compaction plan.
    ///
    /// This method analyzes the dataset's fragments and generates a [`CompactionPlan`]
    /// containing a list of compaction tasks to execute.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Reference to the dataset to be compacted
    async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan>;
}

/// Formulate a plan to compact the files in a dataset
///
/// The compaction plan will contain a list of tasks to execute. Each task
/// will contain approximately `target_rows_per_fragment` rows and will be
/// rewriting fragments that are adjacent in the dataset's fragment list. Some
/// tasks may contain a single fragment when that fragment has deletions that
/// are being materialized and doesn't have any neighbors that need to be
/// compacted.
///
/// Fragments no rewrite task takes are then checked for a column repack
/// against [`Dataset::column_layout_stats`], one task per fragment.
#[derive(Debug, Clone, Default)]
pub struct DefaultCompactionPlanner {
    inner: runtime::DefaultCompactionPlanner,
}

impl DefaultCompactionPlanner {
    pub fn new(options: CompactionOptions) -> Result<Self> {
        Ok(Self {
            inner: runtime::DefaultCompactionPlanner::new(options)?,
        })
    }

    /// Select the publication contract used by this plan. Without an explicit
    /// remapper, planning preserves physical index boundaries. Distributed callers
    /// using [`crate::dataset::index::DatasetIndexRemapperOptions`] at commit may opt into atomic merges.
    ///
    /// ```
    /// use lance::dataset::index::DatasetIndexRemapperOptions;
    /// use lance::dataset::optimize::{CompactionOptions, DefaultCompactionPlanner};
    /// let planner = DefaultCompactionPlanner::new(CompactionOptions::default())?
    ///     .with_index_remapper(&DatasetIndexRemapperOptions::default());
    /// # Ok::<(), lance::Error>(())
    /// ```
    pub fn with_index_remapper(mut self, remapper: &dyn IndexRemapperOptions) -> Self {
        self.inner = self
            .inner
            .with_vector_segment_replacement(remapper.supports_vector_segment_replacement());
        self
    }
}

#[async_trait]
impl CompactionPlanner for DefaultCompactionPlanner {
    async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan> {
        Ok(self.plan_with_budget(dataset).await?.plan)
    }
}

impl DefaultCompactionPlanner {
    /// Plan the compaction of explicitly chosen, mutually disjoint fragment
    /// groups, which need not be adjacent in the manifest. Lance interprets
    /// nothing about why fragments were grouped: the caller chooses the groups
    /// and the output properties, Lance selects and binds the work.
    ///
    /// Inside a group the ordinary rules apply, through the same code as
    /// [`CompactionPlanner::plan`]: a fragment under the target compacts with
    /// its group, one over the target is left alone unless deletions, overlays
    /// or dead columns call for rewriting it by itself, candidates are split at
    /// index boundaries and by output size, and a lone fragment that would
    /// only gain from neighbours is not rewritten. Several small fragments of
    /// one group compact together even when their total stays under the
    /// target. Inputs of different groups are never mixed.
    ///
    /// Outputs replace exactly the selected sources; every other fragment
    /// keeps its id, files and metadata, and the outputs take fresh ids at the
    /// end of the manifest, as any compaction does on this version.
    ///
    /// Refused before anything is written: an empty group, a group out of
    /// ascending id order, a fragment absent from the dataset, named by two
    /// groups, or excluded by [`CompactionOptions::excluded_fragment_ids`].
    /// A member covered by an index this build cannot read is held back
    /// instead of refusing the plan: no metric of it is read, it is left out of
    /// every task, one log line names a bounded sample of the held ids, and
    /// the rest of its group and every other group still plan.
    /// `max_source_fragments`, `max_source_rows` and `max_source_bytes` bound
    /// the selected tasks, in group order. Only rewrites are planned: `scope`
    /// and `max_data_files_per_fragment` are not consulted.
    pub async fn plan_groups(
        &self,
        dataset: &Dataset,
        groups: Vec<CompactionGroup>,
    ) -> Result<CompactionPlan> {
        Ok(self.plan_groups_with_budget(dataset, groups).await?.plan)
    }

    /// Plan caller-supplied groups and report work held back by source budgets.
    ///
    /// Group ordering, output labels, and selected-source validation follow
    /// [`Self::plan_groups`]. Candidate counts include tasks excluded by its
    /// early fragment cutoff. Missing byte sizes in a task already excluded
    /// by that cutoff leave its byte requirement unknown instead of rejecting
    /// the selected prefix.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::optimize::{CompactionGroup, CompactionOptions, DefaultCompactionPlanner};
    /// # async fn plan(dataset: &Dataset, groups: Vec<CompactionGroup>) -> Result<()> {
    /// let result = DefaultCompactionPlanner::new(CompactionOptions {
    ///     max_source_rows: Some(5_000_000), ..Default::default()
    /// })?.plan_groups_with_budget(dataset, groups).await?;
    /// let has_more_work = result.source_budget.budget_exhausted();
    /// # let _ = has_more_work;
    /// # Ok(()) }
    /// ```
    pub async fn plan_groups_with_budget(
        &self,
        dataset: &Dataset,
        groups: Vec<CompactionGroup>,
    ) -> Result<CompactionPlanningResult> {
        let table = dataset.table_view().to_table();
        let bindings = LanceCompactionHooks::new(dataset, None);
        let result = self
            .inner
            .plan_groups_with_budget_and_hooks(&table, groups, &bindings.runtime())
            .await?;
        Ok(CompactionPlanningResult {
            plan: result.plan.into(),
            source_budget: result.source_budget,
        })
    }

    /// Plan compaction and report work held back by the source budgets.
    /// Unlike an empty plan alone, the summary distinguishes an idle dataset
    /// from an oversized first task. Tasks are admitted whole. Inspect
    /// [`SourceBudgetSummary::limiting_budgets`] to determine which limits need
    /// adjusting; a task's row count is a minimum for the row limit only.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::optimize::{CompactionOptions, DefaultCompactionPlanner, SourceBudgetLimit};
    /// # async fn plan(dataset: &Dataset) -> Result<()> {
    /// let result = DefaultCompactionPlanner::new(CompactionOptions {
    ///     max_source_rows: Some(5_000_000), ..Default::default()
    /// })?.plan_with_budget(dataset).await?;
    /// if result.source_budget.is_blocked()
    ///     && result.source_budget.limiting_budgets.contains(&SourceBudgetLimit::Rows)
    /// {
    ///     let minimum = result.source_budget.next_task_source_rows;
    ///     # let _ = minimum;
    /// }
    /// # Ok(()) }
    /// ```
    pub async fn plan_with_budget(&self, dataset: &Dataset) -> Result<CompactionPlanningResult> {
        let table = dataset.table_view().to_table();
        let bindings = LanceCompactionHooks::new(dataset, None);
        let result = self
            .inner
            .plan_with_budget_and_hooks(&table, &bindings.runtime())
            .await?;
        Ok(CompactionPlanningResult {
            plan: result.plan.into(),
            source_budget: result.source_budget,
        })
    }
}

/// Plan compaction with the default candidate and index-boundary rules.
pub async fn plan_compaction(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Result<CompactionPlan> {
    let planner = DefaultCompactionPlanner::new(options.clone())?;
    planner.plan(dataset).await
}
