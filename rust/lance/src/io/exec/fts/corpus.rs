// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Corpus-scorer coordination shared by indexed and flat FTS branches.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, Weak};

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::metrics::{MetricValue, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    ReplaceChildrenOptions,
};
use datafusion_physical_expr::{Distribution, PhysicalExpr};
use datafusion_physical_plan::ExecutionPlanProperties;
use lance_core::utils::futures::FinallyStreamExt;
use lance_index::scalar::inverted::query::Operator;
use lance_index::scalar::inverted::{CombinedFieldsBM25Scorer, MemBM25Scorer};

use super::{
    CombinedFieldsQueryExec, FlatCombinedFieldsExec, FlatMatchQueryExec, MatchQueryExec,
    PhraseQueryExec,
};
use crate::Dataset;

/// The two branches sharing statistics within one [`SharedFtsScorerExec`].
///
/// Paths are child indices relative to the owner's single input, not to the
/// owner itself. They describe the encoded plan tree and contain no process-local
/// scorer identity. Read a new binding after rewriting that tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtsCorpusBinding {
    /// Path to the flat branch that measures and publishes corpus statistics.
    pub producer_path: Vec<usize>,
    /// Path to the indexed branch that consumes those statistics.
    pub consumer_path: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScorerRole {
    Producer,
    Consumer,
}

type SharedScorerResult<S> = std::result::Result<Arc<S>, Arc<str>>;

/// Coordinates BM25 corpus statistics between the indexed and flat branches
/// of a mixed search. The flat branch extends the indexed statistics with the
/// unindexed documents, then publishes the resulting corpus-wide scorer.
///
/// Generic over the scorer so the single-column path can share a
/// [`MemBM25Scorer`] and `combined_fields` a [`CombinedFieldsBM25Scorer`]; the
/// coordination is the same either way.
#[derive(Debug)]
pub struct SharedFtsScorer<S = MemBM25Scorer> {
    sender: tokio::sync::watch::Sender<Option<SharedScorerResult<S>>>,
}

impl<S: Send + Sync + 'static> SharedFtsScorer<S> {
    pub(crate) fn new() -> Self {
        let (sender, _) = tokio::sync::watch::channel(None);
        Self { sender }
    }

    pub(super) fn publish(&self, scorer: Arc<S>) {
        self.sender.send_replace(Some(Ok(scorer)));
    }

    pub(super) fn publish_error(&self, error: &DataFusionError) {
        self.sender
            .send_replace(Some(Err(Arc::from(error.to_string()))));
    }

    pub(super) async fn wait(&self) -> DataFusionResult<Arc<S>> {
        let mut receiver = self.sender.subscribe();
        loop {
            let result = receiver.borrow_and_update().clone();
            if let Some(result) = result {
                return result.map_err(|message| DataFusionError::Execution(message.to_string()));
            }
            receiver.changed().await.map_err(|_| {
                DataFusionError::Execution(
                    "mixed FTS corpus scorer producer stopped before publishing statistics"
                        .to_string(),
                )
            })?;
        }
    }
}

pub(super) struct SharedFtsScorerProducer<S: Send + Sync + 'static = MemBM25Scorer> {
    scorer: Arc<SharedFtsScorer<S>>,
    completed: bool,
}

impl<S: Send + Sync + 'static> SharedFtsScorerProducer<S> {
    pub(super) fn new(scorer: Arc<SharedFtsScorer<S>>) -> Self {
        Self {
            scorer,
            completed: false,
        }
    }

    pub(super) fn publish(mut self, scorer: Arc<S>) {
        self.scorer.publish(scorer);
        self.completed = true;
    }

    pub(super) fn publish_error(mut self, error: &DataFusionError) {
        self.scorer.publish_error(error);
        self.completed = true;
    }
}

impl<S: Send + Sync + 'static> Drop for SharedFtsScorerProducer<S> {
    fn drop(&mut self) {
        if !self.completed {
            self.scorer.sender.send_replace(Some(Err(Arc::from(
                "mixed FTS corpus scorer producer was cancelled before publishing statistics",
            ))));
        }
    }
}

/// Owns one mixed FTS plan's scorer lifecycle. Each execution gets a
/// fresh producer/consumer pair, including after failure or child replacement.
/// Re-execution still requires replayable children: an overlay-stale scalar
/// prefilter can contain a consumable `OneShotExec` that this owner cannot replay.
#[derive(Debug)]
pub struct SharedFtsScorerExec {
    input: Arc<dyn ExecutionPlan>,
    scorer: SharedFtsCorpus,
    /// Retain metric handles, never a finished execution's task-owning plan.
    /// Concurrent executions have independent scorers. Rebound metrics describe
    /// the latest execution; template-retained counters keep their existing semantics.
    metrics: Arc<Mutex<SharedFtsExecutionMetrics>>,
}

#[derive(Debug, Clone)]
enum SharedFtsCorpus {
    Match(Arc<SharedFtsScorer>),
    Combined(Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>),
}

impl SharedFtsCorpus {
    fn role(&self, plan: &dyn ExecutionPlan) -> Option<ScorerRole> {
        match self {
            Self::Match(template) => {
                if let Some(node) = plan.downcast_ref::<MatchQueryExec>()
                    && node.uses_shared_scorer(template)
                {
                    return Some(ScorerRole::Consumer);
                }
                if let Some(node) = plan.downcast_ref::<PhraseQueryExec>()
                    && node.uses_shared_scorer(template)
                {
                    return Some(ScorerRole::Consumer);
                }
                if let Some(node) = plan.downcast_ref::<FlatMatchQueryExec>()
                    && node.uses_shared_scorer(template)
                {
                    return Some(ScorerRole::Producer);
                }
            }
            Self::Combined(template) => {
                if let Some(node) = plan.downcast_ref::<CombinedFieldsQueryExec>()
                    && node.uses_shared_scorer(template)
                {
                    return Some(ScorerRole::Consumer);
                }
                if let Some(node) = plan.downcast_ref::<FlatCombinedFieldsExec>()
                    && node.uses_shared_scorer(template)
                {
                    return Some(ScorerRole::Producer);
                }
            }
        }
        None
    }

    fn bind_leaf(
        &self,
        plan: &dyn ExecutionPlan,
        role: ScorerRole,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let wrong_role = || {
            DataFusionError::Plan(format!(
                "FTS corpus {role:?} cannot bind to {}",
                plan.name()
            ))
        };
        let rebound: Arc<dyn ExecutionPlan> = match (self, role) {
            (Self::Match(scorer), ScorerRole::Consumer) => {
                if let Some(node) = plan.downcast_ref::<MatchQueryExec>() {
                    node.rebuilt_with_shared_scorer(scorer.clone())
                } else {
                    plan.downcast_ref::<PhraseQueryExec>()
                        .ok_or_else(wrong_role)?
                        .rebuilt_with_shared_scorer(scorer.clone())
                }
            }
            (Self::Match(scorer), ScorerRole::Producer) => plan
                .downcast_ref::<FlatMatchQueryExec>()
                .ok_or_else(wrong_role)?
                .rebuilt_with_shared_scorer(scorer.clone()),
            (Self::Combined(scorer), ScorerRole::Consumer) => Arc::new(
                plan.downcast_ref::<CombinedFieldsQueryExec>()
                    .ok_or_else(wrong_role)?
                    .clone()
                    .with_shared_scorer(scorer.clone()),
            ),
            (Self::Combined(scorer), ScorerRole::Producer) => Arc::new(
                plan.downcast_ref::<FlatCombinedFieldsExec>()
                    .ok_or_else(wrong_role)?
                    .clone()
                    .with_shared_scorer(scorer.clone()),
            ),
        };
        Ok(rebound)
    }

    fn for_pair(
        producer: &dyn ExecutionPlan,
        consumer: &dyn ExecutionPlan,
    ) -> DataFusionResult<Self> {
        if let (Some(producer), Some(consumer)) = (
            producer.downcast_ref::<FlatMatchQueryExec>(),
            consumer.downcast_ref::<MatchQueryExec>(),
        ) {
            require_same_snapshot(producer.dataset(), consumer.dataset())?;
            if producer.query() != consumer.query()
                || producer.document_granularity() != consumer.document_granularity()
            {
                return Err(DataFusionError::Plan(
                    "FTS corpus branches must use the same Match query and document granularity"
                        .to_string(),
                ));
            }
            return Ok(Self::Match(Arc::new(SharedFtsScorer::new())));
        }
        if let (Some(producer), Some(consumer)) = (
            producer.downcast_ref::<FlatMatchQueryExec>(),
            consumer.downcast_ref::<PhraseQueryExec>(),
        ) {
            require_same_snapshot(producer.dataset(), consumer.dataset())?;
            if producer.query().terms != consumer.query().terms
                || producer.query().column != consumer.query().column
                || producer.query().operator != Operator::And
                || producer.query().fuzziness != Some(0)
                || producer.query().boost != 1.0
                || producer.document_granularity() != consumer.document_granularity()
                || producer.params().phrase_slop != consumer.params().phrase_slop
            {
                return Err(DataFusionError::Plan(
                    "FTS corpus phrase branches must use an exact AND match of the same terms and column with the same slop and document granularity".to_string(),
                ));
            }
            return Ok(Self::Match(Arc::new(SharedFtsScorer::new())));
        }
        if let (Some(producer), Some(consumer)) = (
            producer.downcast_ref::<FlatCombinedFieldsExec>(),
            consumer.downcast_ref::<CombinedFieldsQueryExec>(),
        ) {
            require_same_snapshot(producer.dataset(), consumer.dataset())?;
            if producer.query() != consumer.query() {
                return Err(DataFusionError::Plan(
                    "FTS corpus branches must use the same combined_fields query".to_string(),
                ));
            }
            return Ok(Self::Combined(Arc::new(SharedFtsScorer::new())));
        }
        Err(DataFusionError::Plan(format!(
            "FTS corpus requires a matching flat producer and indexed consumer, got {} and {}",
            producer.name(),
            consumer.name(),
        )))
    }
}

fn require_same_snapshot(producer: &Dataset, consumer: &Dataset) -> DataFusionResult<()> {
    // A URI/version pair does not distinguish in-memory manifest views or
    // different object-store contexts. A shallow Dataset clone preserves these
    // handles; a decoder resolves the pinned table once for the whole corpus.
    if producer.uri() != consumer.uri()
        || !Arc::ptr_eq(&producer.manifest, &consumer.manifest)
        || !Arc::ptr_eq(&producer.object_store, &consumer.object_store)
    {
        return Err(DataFusionError::Plan(format!(
            "FTS corpus branches must share an opened dataset snapshot (producer version {}, consumer version {})",
            producer.version_id(),
            consumer.version_id(),
        )));
    }
    Ok(())
}

fn has_shared_scorer(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<MatchQueryExec>()
        .is_some_and(|node| node.has_shared_scorer())
        || plan
            .downcast_ref::<FlatMatchQueryExec>()
            .is_some_and(|node| node.has_shared_scorer())
        || plan
            .downcast_ref::<CombinedFieldsQueryExec>()
            .is_some_and(|node| node.has_shared_scorer())
        || plan
            .downcast_ref::<FlatCombinedFieldsExec>()
            .is_some_and(|node| node.has_shared_scorer())
        || plan
            .downcast_ref::<PhraseQueryExec>()
            .is_some_and(|node| node.has_shared_scorer())
}

/// Visit only this owner's scope, leaving nested corpus owners opaque.
fn walk_scope(
    plan: &Arc<dyn ExecutionPlan>,
    path: &mut Vec<usize>,
    visit: &mut impl FnMut(&[usize], &Arc<dyn ExecutionPlan>) -> DataFusionResult<()>,
) -> DataFusionResult<()> {
    if plan.is::<SharedFtsScorerExec>() {
        return Ok(());
    }
    visit(path, plan)?;
    for (index, child) in plan.children().into_iter().enumerate() {
        path.push(index);
        walk_scope(child, path, visit)?;
        path.pop();
    }
    Ok(())
}

fn at_path<'a>(
    input: &'a Arc<dyn ExecutionPlan>,
    path: &[usize],
) -> DataFusionResult<&'a Arc<dyn ExecutionPlan>> {
    let mut node = input;
    for (depth, &index) in path.iter().enumerate() {
        if node.is::<SharedFtsScorerExec>() {
            return Err(DataFusionError::Plan(format!(
                "FTS corpus binding path {path:?} crosses a nested owner at depth {depth}",
            )));
        }
        node = node.children().get(index).copied().ok_or_else(|| {
            DataFusionError::Plan(format!(
                "FTS corpus binding path {path:?} has no child {index} at depth {depth}",
            ))
        })?;
    }
    Ok(node)
}

fn bind_paths(
    plan: Arc<dyn ExecutionPlan>,
    path: &mut Vec<usize>,
    binding: &FtsCorpusBinding,
    scorer: &SharedFtsCorpus,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    if path.as_slice() == binding.producer_path.as_slice() {
        return scorer.bind_leaf(plan.as_ref(), ScorerRole::Producer);
    }
    if path.as_slice() == binding.consumer_path.as_slice() {
        return scorer.bind_leaf(plan.as_ref(), ScorerRole::Consumer);
    }
    if plan.is::<SharedFtsScorerExec>() {
        return Ok(plan);
    }
    let children = plan.children();
    let mut has_changes = false;
    let mut rebound = Vec::with_capacity(children.len());
    for (index, child) in children.into_iter().enumerate() {
        path.push(index);
        let next = bind_paths(child.clone(), path, binding, scorer)?;
        path.pop();
        has_changes |= !Arc::ptr_eq(child, &next);
        rebound.push(next);
    }
    if has_changes {
        plan.replace_children(
            rebound,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    } else {
        Ok(plan)
    }
}

#[derive(Debug)]
struct SharedFtsExecutionMetrics {
    execution: Arc<()>,
    active: Option<Weak<dyn ExecutionPlan>>,
    snapshot: MetricsSet,
}

struct SharedFtsMetricsGuard {
    input: Arc<dyn ExecutionPlan>,
    template: Arc<dyn ExecutionPlan>,
    metrics: Arc<Mutex<SharedFtsExecutionMetrics>>,
    execution: Arc<()>,
}

impl Drop for SharedFtsMetricsGuard {
    fn drop(&mut self) {
        let snapshot =
            SharedFtsScorerExec::execution_metrics(self.template.as_ref(), self.input.as_ref());
        match self.metrics.lock() {
            Ok(mut metrics) if Arc::ptr_eq(&metrics.execution, &self.execution) => {
                metrics.snapshot = snapshot;
                metrics.active = None;
            }
            Ok(_) => {}
            Err(error) => log::warn!("could not retain FTS runtime metrics: {error}"),
        }
    }
}

impl SharedFtsScorerExec {
    /// Reconnect the producer and consumer in a decoded native corpus plan.
    ///
    /// The binding refers to child paths in `input`. The selected leaves must be
    /// a FlatMatch/Match, FlatMatch/Phrase, or FlatCombinedFields/CombinedFields
    /// pair with the same query and opened dataset snapshot. Resolve the dataset
    /// once and reuse it (or a shallow clone) for both leaves. Candidate filters
    /// and limits may differ; their separation from the statistics corpus is
    /// preserved.
    /// Native segment/fragment scopes and scorer overrides remain authoritative.
    ///
    /// The two branches must be separate and cannot cross another corpus owner.
    /// Other shared-scorer leaves require their own nested owner. This creates
    /// fresh template coordination, never restoring a scorer from the encoding
    /// process. Each execution then gets fresh state through the native lifecycle.
    /// Children must still be replayable for re-execution.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use datafusion::physical_plan::ExecutionPlan;
    /// # use datafusion::error::Result;
    /// # use lance::io::exec::fts::SharedFtsScorerExec;
    /// # fn rebuild(owner: &SharedFtsScorerExec, decoded_input: Arc<dyn ExecutionPlan>)
    /// #     -> Result<SharedFtsScorerExec> {
    /// let binding = owner.binding()?;
    /// SharedFtsScorerExec::try_new(decoded_input, binding)
    /// # }
    /// ```
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        binding: FtsCorpusBinding,
    ) -> DataFusionResult<Self> {
        if binding.producer_path.starts_with(&binding.consumer_path)
            || binding.consumer_path.starts_with(&binding.producer_path)
        {
            return Err(DataFusionError::Plan(
                "FTS corpus producer and consumer must be in separate branches".to_string(),
            ));
        }
        let producer = at_path(&input, &binding.producer_path)?;
        let consumer = at_path(&input, &binding.consumer_path)?;
        let scorer = SharedFtsCorpus::for_pair(producer.as_ref(), consumer.as_ref())?;
        walk_scope(&input, &mut Vec::new(), &mut |path, node| {
            if path != binding.producer_path.as_slice()
                && path != binding.consumer_path.as_slice()
                && has_shared_scorer(node.as_ref())
            {
                return Err(DataFusionError::Plan(format!(
                    "FTS corpus has an unbound shared-scorer leaf at path {path:?}",
                )));
            }
            Ok(())
        })?;
        let input = bind_paths(input, &mut Vec::new(), &binding, &scorer)?;
        Ok(Self::with_corpus(input, scorer))
    }

    /// Locate this owner's producer and consumer in its current input tree.
    ///
    /// Encode these paths with that input, then pass the decoded children and
    /// binding to [`Self::try_new`]. Nested owners have independent bindings.
    /// The returned value contains no process-local identity or computed scores.
    pub fn binding(&self) -> DataFusionResult<FtsCorpusBinding> {
        let mut producers = Vec::new();
        let mut consumers = Vec::new();
        walk_scope(&self.input, &mut Vec::new(), &mut |path, node| {
            match self.scorer.role(node.as_ref()) {
                Some(ScorerRole::Producer) => producers.push(path.to_vec()),
                Some(ScorerRole::Consumer) => consumers.push(path.to_vec()),
                None if has_shared_scorer(node.as_ref()) => {
                    return Err(DataFusionError::Plan(format!(
                        "FTS corpus has an unbound shared-scorer leaf at path {path:?}",
                    )));
                }
                None => {}
            }
            Ok(())
        })?;
        if producers.len() != 1 || consumers.len() != 1 {
            return Err(DataFusionError::Plan(format!(
                "FTS corpus binding requires one producer and consumer, got {} and {}",
                producers.len(),
                consumers.len(),
            )));
        }
        Ok(FtsCorpusBinding {
            producer_path: producers.remove(0),
            consumer_path: consumers.remove(0),
        })
    }

    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, scorer: Arc<SharedFtsScorer>) -> Self {
        Self::with_corpus(input, SharedFtsCorpus::Match(scorer))
    }

    pub(crate) fn new_combined(
        input: Arc<dyn ExecutionPlan>,
        scorer: Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>,
    ) -> Self {
        Self::with_corpus(input, SharedFtsCorpus::Combined(scorer))
    }

    fn with_corpus(input: Arc<dyn ExecutionPlan>, scorer: SharedFtsCorpus) -> Self {
        Self {
            metrics: Arc::new(Mutex::new(SharedFtsExecutionMetrics {
                execution: Arc::new(()),
                active: None,
                snapshot: MetricsSet::new(),
            })),
            input,
            scorer,
        }
    }

    fn prepare_execution(&self) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let fresh = match &self.scorer {
            SharedFtsCorpus::Match(_) => SharedFtsCorpus::Match(Arc::new(SharedFtsScorer::new())),
            SharedFtsCorpus::Combined(_) => {
                SharedFtsCorpus::Combined(Arc::new(SharedFtsScorer::new()))
            }
        };
        let mut consumers = 0;
        let mut producers = 0;
        // Reset other execution state too, particularly a previous top-k sort's
        // score threshold. Each execution owns its corpus independently.
        let input = reset_plan_states(self.input.clone())?;
        let input = input
            .transform_down_up(
                |plan| {
                    // An owner cloned by a child rewrite can share this
                    // template Arc, but still owns its own execution scope.
                    let recursion = if plan.is::<Self>() {
                        TreeNodeRecursion::Jump
                    } else {
                        TreeNodeRecursion::Continue
                    };
                    Ok(Transformed::new(plan, false, recursion))
                },
                |plan| {
                    let Some(role) = self.scorer.role(plan.as_ref()) else {
                        return Ok(Transformed::no(plan));
                    };
                    match role {
                        ScorerRole::Producer => producers += 1,
                        ScorerRole::Consumer => consumers += 1,
                    }
                    Ok(Transformed::yes(fresh.bind_leaf(plan.as_ref(), role)?))
                },
            )?
            .data;
        if consumers != 1 || producers != 1 {
            return Err(DataFusionError::Internal(format!(
                "FTS corpus requires one scorer consumer and producer, got {consumers} and {producers}"
            )));
        }
        Ok(input)
    }

    fn execution_metrics(template: &dyn ExecutionPlan, runtime: &dyn ExecutionPlan) -> MetricsSet {
        fn collect(plan: &dyn ExecutionPlan, metrics: &mut MetricsSet, only_named: bool) {
            if let Some(node_metrics) = plan.metrics() {
                for metric in node_metrics.iter() {
                    if !only_named
                        || matches!(
                            metric.value(),
                            MetricValue::Count { .. }
                                | MetricValue::Gauge { .. }
                                | MetricValue::Time { .. }
                                | MetricValue::PruningMetrics { .. }
                                | MetricValue::Ratio { .. }
                                | MetricValue::Custom { .. }
                        )
                    {
                        metrics.push(metric.clone());
                    }
                }
            }
            for child in plan.children() {
                collect(child.as_ref(), metrics, only_named);
            }
        }
        let mut template_metrics = MetricsSet::new();
        collect(template, &mut template_metrics, false);
        let mut seen = template_metrics
            .iter()
            .map(|metric| Arc::as_ptr(metric) as usize)
            .collect::<HashSet<_>>();
        // The wrapper has its input's output and compute metrics. Descendant
        // baseline metrics describe different operators, so only their named
        // I/O, scorer and other diagnostic metrics belong in this aggregation.
        let mut runtime_metrics = runtime.metrics().unwrap_or_default();
        for child in runtime.children() {
            collect(child.as_ref(), &mut runtime_metrics, true);
        }
        let mut metrics = MetricsSet::new();
        for metric in runtime_metrics.iter() {
            // Some leaf resets intentionally retain metric handles. The visible
            // template already exposes these, so do not count them twice.
            if seen.insert(Arc::as_ptr(metric) as usize) {
                metrics.push(metric.clone());
            }
        }
        metrics
    }
}

impl DisplayAs for SharedFtsScorerExec {
    fn fmt_as(&self, _format: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match &self.scorer {
            SharedFtsCorpus::Match(_) => "MatchCorpus",
            SharedFtsCorpus::Combined(_) => "CombinedFieldsCorpus",
        })
    }
}

impl ExecutionPlan for SharedFtsScorerExec {
    fn name(&self) -> &str {
        "SharedFtsScorerExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "FTS corpus requires one input, got {}",
                children.len()
            )));
        }
        let input = children
            .pop()
            .ok_or_else(|| DataFusionError::Internal("FTS corpus input is missing".to_string()))?;
        Ok(Arc::new(Self::with_corpus(input, self.scorer.clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "FTS corpus requires partition 0, got {partition}"
            )));
        }
        let input = self.prepare_execution()?;
        if input.output_partitioning().partition_count() != 1 {
            return Err(DataFusionError::Internal(
                "FTS corpus input must have one output partition".to_string(),
            ));
        }
        let execution = Arc::new(());
        let result = input.execute(partition, context);
        let snapshot = Self::execution_metrics(self.input.as_ref(), input.as_ref());
        *self.metrics.lock().map_err(|_| {
            DataFusionError::Internal("FTS runtime lock was poisoned".to_string())
        })? = SharedFtsExecutionMetrics {
            execution: execution.clone(),
            active: Some(Arc::downgrade(&input)),
            snapshot,
        };
        let stream = result?;
        let schema = stream.schema();
        let guard = SharedFtsMetricsGuard {
            input,
            template: self.input.clone(),
            metrics: self.metrics.clone(),
            execution,
        };
        // The captured guard also runs when this closure is dropped before EOF,
        // preserving lazily registered metrics on error or cancellation.
        let stream = stream.finally(move || drop(guard));
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        let (active, snapshot) = {
            let metrics = self.metrics.lock().ok()?;
            (
                metrics.active.as_ref().and_then(Weak::upgrade),
                metrics.snapshot.clone(),
            )
        };
        // Expose active metrics before EOF without retaining the execution
        // between inspections.
        Some(match active {
            Some(input) => Self::execution_metrics(self.input.as_ref(), input.as_ref()),
            None => snapshot,
        })
    }
}

#[cfg(test)]
mod tests {
    mod reconstruction;
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use arrow_array::record_batch;
    use arrow_array::{Float32Array, RecordBatch, UInt64Array};
    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::physical_plan::union::UnionExec;
    use datafusion::physical_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};
    use futures::{FutureExt, TryStreamExt};
    use lance_core::ROW_ID;
    use lance_datafusion::exec::{ExecutionSummaryCounts, collect_execution_metrics};
    use lance_datafusion::utils::PARTITIONS_SEARCHED_METRIC;
    use lance_index::scalar::inverted::query::{FtsSearchParams, MatchQuery};
    use lance_index::scalar::inverted::{DocumentGranularity, FTS_SCHEMA, SCORE_COL};
    use rstest::rstest;

    use crate::io::exec::PreFilterSource;
    use crate::io::exec::fts::tests::{
        FailingResidualExec, ResidualFailure, candidate_input, indexed_match_dataset, memory_input,
        metric_value,
    };

    #[rstest]
    #[case::execute(ResidualFailure::Execute)]
    #[case::stream(ResidualFailure::Stream)]
    #[case::cancel(ResidualFailure::Cancel)]
    #[tokio::test]
    async fn test_9058_shared_corpus_retries_after_failure(
        #[case] failure: ResidualFailure,
        #[values(false, true)] is_reconstructed: bool,
    ) {
        let dataset = indexed_match_dataset().await;
        let query = MatchQuery::new("alpha beta".to_string())
            .with_column(Some("text".to_string()))
            .with_document_granularity(DocumentGranularity::Row);
        let residual = Arc::new(FailingResidualExec::new(
            memory_input(
                record_batch!(
                    ("text", Utf8, vec!["alpha"; 10]),
                    (ROW_ID, UInt64, (100..110).collect::<Vec<u64>>())
                )
                .unwrap(),
            ),
            failure,
        ));
        let shared = Arc::new(SharedFtsScorer::new());
        let indexed = Arc::new(
            MatchQueryExec::new(
                dataset.clone(),
                query.clone(),
                FtsSearchParams::default().with_limit(Some(1)),
                PreFilterSource::None,
            )
            .unwrap()
            .with_shared_scorer(shared.clone()),
        );
        let flat = Arc::new(
            FlatMatchQueryExec::new(
                dataset.clone(),
                query.clone(),
                FtsSearchParams::default(),
                residual.clone(),
            )
            .unwrap()
            .with_shared_scorer(shared.clone())
            .with_candidate_filter(candidate_input(vec![], false)),
        );
        let union = UnionExec::try_new(vec![indexed, flat]).unwrap();
        let input: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(union));
        let owner = Arc::new(SharedFtsScorerExec::new(input, shared));
        let owner: Arc<dyn ExecutionPlan> = if is_reconstructed {
            let binding = owner.binding().unwrap();
            // Model a decoder: neither leaf carries the original shared Arc,
            // and a shallow Dataset clone is still the same opened snapshot.
            let indexed = Arc::new(
                MatchQueryExec::new(
                    dataset.clone(),
                    query.clone(),
                    FtsSearchParams::default().with_limit(Some(1)),
                    PreFilterSource::None,
                )
                .unwrap(),
            );
            let flat = Arc::new(
                FlatMatchQueryExec::new(
                    Arc::new(dataset.as_ref().clone()),
                    query,
                    FtsSearchParams::default(),
                    residual.clone(),
                )
                .unwrap()
                .with_candidate_filter(candidate_input(vec![], false)),
            );
            let decoded: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(
                UnionExec::try_new(vec![indexed, flat]).unwrap(),
            ));
            let rebuilt = SharedFtsScorerExec::try_new(decoded, binding.clone()).unwrap();
            assert_eq!(rebuilt.binding().unwrap(), binding);
            Arc::new(rebuilt)
        } else {
            owner
        };
        let input = owner.children()[0].clone();
        let owner = owner
            .replace_children(
                vec![input],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap();
        let distributions = owner.input_distribution_requirements();
        let distributions = distributions.per_child_distributions().collect::<Vec<_>>();
        assert_eq!(distributions.len(), 1);
        assert!(matches!(
            distributions[0],
            datafusion_physical_expr::Distribution::SinglePartition
        ));
        assert_eq!(owner.schema(), FTS_SCHEMA.clone());
        let context = Arc::new(TaskContext::default());
        let stream = owner.execute(0, context.clone()).unwrap();
        if matches!(failure, ResidualFailure::Cancel) {
            tokio::time::timeout(Duration::from_secs(1), residual.started.notified())
                .await
                .unwrap();
            drop(stream);
            tokio::time::timeout(Duration::from_secs(1), async {
                while !residual.dropped.load(Ordering::SeqCst) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect(
                "dropping the output must cancel the residual stream while the owner stays alive",
            );
        } else {
            let error =
                tokio::time::timeout(Duration::from_secs(1), stream.try_collect::<Vec<_>>())
                    .await
                    .unwrap()
                    .unwrap_err();
            assert!(error.to_string().contains("injected residual"), "{error}");
        }
        let mut counts = ExecutionSummaryCounts::default();
        collect_execution_metrics(owner.as_ref(), &mut counts);
        assert_eq!(counts.all_counts["residual_executions"], 1);
        assert!(
            owner
                .metrics()
                .unwrap()
                .iter()
                .any(|metric| metric.value().name() == "scorer_build_ms")
        );

        // Poll the next execution's real indexed consumer through index opening
        // before starting its producer. It must wait, never see the old error.
        let runtime = owner
            .downcast_ref::<SharedFtsScorerExec>()
            .unwrap()
            .prepare_execution()
            .unwrap();
        let union = runtime.children()[0].clone();
        let indexed = union.children()[0].clone();
        let flat = union.children()[1].clone();
        let mut indexed_stream = indexed.execute(0, context.clone()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                tokio::select! {
                    result = indexed_stream.try_next() => panic!("consumer completed before its producer: {result:?}"),
                    _ = tokio::task::yield_now() => {}
                }
                if metric_value(indexed.as_ref(), PARTITIONS_SEARCHED_METRIC) > 0 { break; }
            }
        }).await.unwrap();
        assert!(indexed_stream.try_next().now_or_never().is_none());
        let flat_batches: Vec<_> = flat
            .execute(0, context.clone())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(
            flat_batches
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            0
        );
        let result = tokio::time::timeout(Duration::from_secs(1), indexed_stream.try_next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            result[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            1
        );

        let mut stream = owner.execute(0, context).unwrap();
        let result = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let batch = stream.try_next().await.unwrap().unwrap();
                if batch.num_rows() > 0 {
                    break batch;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(result.num_rows(), 1);
        assert_eq!(
            result[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            1
        );
        assert!(
            (result[SCORE_COL]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(0)
                - 2.1594842)
                .abs()
                < 1e-5
        );
        let mut counts = ExecutionSummaryCounts::default();
        // Inspect metrics before EOF or stream drop. Lazy registrations must
        // already be visible while the owner is still executing.
        collect_execution_metrics(owner.as_ref(), &mut counts);
        assert_eq!(
            owner.metrics().unwrap().output_rows(),
            Some(result.num_rows())
        );
        assert_eq!(counts.all_counts[PARTITIONS_SEARCHED_METRIC], 2);
        assert_eq!(
            counts.all_counts["residual_executions"], 3,
            "shared source metrics must not be counted twice"
        );
        assert!(owner.metrics().unwrap().elapsed_compute().unwrap() > 0);
        drop(stream);
        let mut final_counts = ExecutionSummaryCounts::default();
        collect_execution_metrics(owner.as_ref(), &mut final_counts);
        assert_eq!(final_counts.all_counts, counts.all_counts);
    }
}
