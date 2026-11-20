// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scopes of observed operations and of their stages.

use std::sync::Arc;

use crate::Result;

use super::observer::{MemoryObserver, StageWindow};
use super::report::OperationReport;

/// An operation Lance observes, reported as its `operation` label.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObservedOperation {
    /// Building one index, through `create_index`.
    CreateIndex,
    /// Joining the segments of an index built across workers, through
    /// `merge_index_metadata`.
    MergeIndexMetadata,
    /// Compacting fragments, in one process or as distributed tasks and their
    /// commit.
    Compaction,
    /// Optimizing indices, in one process or as a distributed plan, its tasks
    /// and its merges.
    OptimizeIndices,
    /// Flushing a MemWAL memtable to storage, once per attempt.
    MemWalFlush,
    /// Writing new data: creating, appending to or overwriting a dataset, or
    /// writing fragments to commit later.
    Write,
    /// Merging new data into a dataset with `merge_insert`, over all its
    /// attempts.
    MergeInsert,
    /// Updating rows, over all attempts.
    Update,
    /// Deleting rows, over all attempts.
    Delete,
}

impl ObservedOperation {
    /// The `operation` label of the operation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CreateIndex => "create_index",
            Self::MergeIndexMetadata => "merge_index_metadata",
            Self::Compaction => "compaction",
            Self::OptimizeIndices => "optimize_indices",
            Self::MemWalFlush => "mem_wal_flush",
            Self::Write => "write",
            Self::MergeInsert => "merge_insert",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
}

/// An operation whose memory is observed while observation is on (see
/// [`set_observation`](super::set_observation)).
///
/// The public entry point of the operation opens the scope, and
/// [`Self::finish`] closes it once the operation has succeeded: after its
/// commit when it commits. Work nested in the operation reports its stages
/// through [`Self::stages`] and never opens a scope of its own. A scope dropped
/// without `finish` never reports its completion, which is how a failed,
/// cancelled or killed operation shows.
///
/// ```
/// # use lance_core::memory::{ObservedOperation, OperationScope};
/// # fn example() -> lance_core::Result<()> {
/// let scope = OperationScope::start(ObservedOperation::Compaction, || None)?;
/// let stage = scope.stages().start("rewrite_files");
/// // ... the work of the stage ...
/// stage.complete();
/// scope.finish();
/// # Ok(())
/// # }
/// ```
#[doc(hidden)]
#[must_use = "an operation dropped without `finish` never reports its completion"]
#[derive(Debug)]
pub struct OperationScope {
    observed: Option<(StageWindow, Arc<Observed>)>,
}

#[derive(Debug)]
struct Observed {
    observer: Arc<MemoryObserver>,
    report: OperationReport,
}

impl OperationScope {
    /// Observes the operation with the process-wide observer when
    /// observation is on (see [`set_observation`](super::set_observation));
    /// `index_type` is only evaluated then. It names the type of the index the
    /// operation builds, and is `None` for operations that build no single
    /// index.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when [`OBSERVE_ENV_VAR`](super::OBSERVE_ENV_VAR)
    /// holds something other than a boolean and `set_observation` has not been called.
    pub fn start(
        operation: ObservedOperation,
        index_type: impl FnOnce() -> Option<String>,
    ) -> Result<Self> {
        Ok(match MemoryObserver::global()? {
            Some(observer) => Self::with_observer(observer, operation, index_type()),
            None => Self::unobserved(),
        })
    }

    /// Observes the operation with `observer`.
    pub fn with_observer(
        observer: Arc<MemoryObserver>,
        operation: ObservedOperation,
        index_type: Option<String>,
    ) -> Self {
        let window = observer.open_window();
        let report = OperationReport::for_operation(operation.as_str(), index_type);
        report.start();
        Self {
            observed: Some((window, Arc::new(Observed { observer, report }))),
        }
    }

    /// An operation that is not observed: it and its stages report nothing.
    pub fn unobserved() -> Self {
        Self { observed: None }
    }

    /// The reporter that opens the stages of the operation, for this scope and
    /// for the work nested in it.
    pub fn stages(&self) -> StageReporter {
        StageReporter(self.observed.as_ref().map(|(_, observed)| observed.clone()))
    }

    /// Reports the peak memory of the whole operation, once it has succeeded.
    /// Stages still open are not completed by this.
    pub fn finish(self) {
        if let Some((window, observed)) = self.observed {
            observed.report.complete(window.close());
        }
    }
}

/// Opens stages of an observed operation; every clone reports into the same
/// operation.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct StageReporter(Option<Arc<Observed>>);

impl StageReporter {
    /// Opens a stage of the operation, a no-op when the operation is not
    /// observed. The stage covers the work until [`Stage::complete`]; open it
    /// before its work starts, which for lazy streams means before they are
    /// first polled.
    pub fn start(&self, stage: &str) -> Stage {
        Stage(self.0.as_ref().map(|observed| {
            let window = observed.observer.open_window();
            observed.report.stage_start(stage);
            (window, observed.clone(), stage.to_string())
        }))
    }

    /// Whether stages are reported, that is whether the operation is observed.
    pub fn is_observed(&self) -> bool {
        self.0.is_some()
    }
}

/// A stage of an observed operation, from [`StageReporter::start`] to
/// [`Self::complete`]. A stage dropped without `complete` never reports its
/// completion.
#[doc(hidden)]
#[must_use = "a stage dropped without `complete` never reports its completion"]
#[derive(Debug)]
pub struct Stage(Option<(StageWindow, Arc<Observed>, String)>);

impl Stage {
    /// Reports the peak memory of the stage, once its work has succeeded.
    pub fn complete(self) {
        if let Some((window, observed, stage)) = self.0 {
            observed.report.stage_complete(&stage, window.close());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::BTreeMap;
    use std::sync::{Mutex, PoisonError};
    use std::time::Duration;

    use super::*;
    use crate::memory::{EMITTING, OBSERVE_ENV_VAR};
    use crate::utils::tracing::TRACE_MEMORY;

    fn observer() -> Arc<MemoryObserver> {
        Arc::new(MemoryObserver::new(Duration::from_millis(1)))
    }

    #[derive(Default)]
    struct Fields(BTreeMap<&'static str, String>);

    /// Records the fields of every memory event; the memory readings are left
    /// out, since they change while a test runs.
    struct Recorder(Arc<Mutex<Vec<Fields>>>);

    impl tracing::field::Visit for Fields {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.0.insert(field.name(), value.to_string());
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if !field.name().ends_with("_bytes") && field.name() != "pid" {
                self.0.insert(field.name(), format!("{value:?}"));
            }
        }
    }

    impl tracing::Subscriber for Recorder {
        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            metadata.target() == TRACE_MEMORY
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            let mut fields = Fields::default();
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields);
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    /// The memory events `run` emits, each as `event operation [stage]`, and
    /// the distinct operation ids among them.
    fn events_of(run: impl FnOnce()) -> (Vec<String>, Vec<String>) {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::with_default(Recorder(recorded.clone()), run);
        let recorded = recorded.lock().unwrap();
        let events = recorded
            .iter()
            .map(|fields| {
                ["event", "operation", "index_type", "stage"]
                    .iter()
                    .filter_map(|name| fields.0.get(name).map(|value| format!("{name}={value}")))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        let mut ids: Vec<String> = recorded
            .iter()
            .map(|fields| fields.0.get("operation_id").cloned().unwrap_or_default())
            .collect();
        ids.dedup();
        (events, ids)
    }

    #[test]
    fn test_stages_of_every_clone_report_into_one_operation() {
        let _emitting = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        let (events, ids) = events_of(|| {
            let scope =
                OperationScope::with_observer(observer(), ObservedOperation::Compaction, None);
            let stages = scope.stages();
            let nested = stages.clone();
            let rewrite = stages.start("rewrite_files");
            let remap = nested.start("remap_indices");
            remap.complete();
            rewrite.complete();
            scope.finish();
        });
        assert_eq!(ids.len(), 1, "{ids:?}");
        assert_eq!(
            events,
            [
                "event=operation_start operation=compaction",
                "event=stage_start operation=compaction stage=rewrite_files",
                "event=stage_start operation=compaction stage=remap_indices",
                "event=stage_complete operation=compaction stage=remap_indices",
                "event=stage_complete operation=compaction stage=rewrite_files",
                "event=operation_complete operation=compaction",
            ]
        );
    }

    #[test]
    fn test_an_operation_of_one_index_names_its_type_on_every_event() {
        let _emitting = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        let (events, _) = events_of(|| {
            let scope = OperationScope::with_observer(
                observer(),
                ObservedOperation::CreateIndex,
                Some("btree".to_string()),
            );
            scope.stages().start("load_data").complete();
            scope.finish();
        });
        assert_eq!(
            events,
            [
                "event=operation_start operation=create_index index_type=btree",
                "event=stage_start operation=create_index index_type=btree stage=load_data",
                "event=stage_complete operation=create_index index_type=btree stage=load_data",
                "event=operation_complete operation=create_index index_type=btree",
            ]
        );
    }

    #[test]
    fn test_dropped_scopes_and_stages_report_only_their_start() {
        let _emitting = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        let (events, ids) = events_of(|| {
            let scope =
                OperationScope::with_observer(observer(), ObservedOperation::MemWalFlush, None);
            let stages = scope.stages();
            stages.start("write_fragments").complete();
            drop(stages.start("commit"));
            drop(scope);
        });
        assert_eq!(ids.len(), 1, "{ids:?}");
        assert_eq!(
            events,
            [
                "event=operation_start operation=mem_wal_flush",
                "event=stage_start operation=mem_wal_flush stage=write_fragments",
                "event=stage_complete operation=mem_wal_flush stage=write_fragments",
                "event=stage_start operation=mem_wal_flush stage=commit",
            ]
        );
    }

    #[test]
    fn test_an_unobserved_scope_reports_nothing() {
        assert!(
            std::env::var_os(OBSERVE_ENV_VAR).is_none(),
            "this test needs {OBSERVE_ENV_VAR} unset"
        );
        let _emitting = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        let evaluated = Cell::new(false);
        let (events, _) = events_of(|| {
            let scope = OperationScope::start(ObservedOperation::OptimizeIndices, || {
                evaluated.set(true);
                None
            })
            .unwrap();
            let stages = scope.stages();
            assert!(!stages.is_observed());
            stages.start("plan").complete();
            drop(stages.start("commit"));
            scope.finish();
        });
        assert!(events.is_empty(), "{events:?}");
        assert!(!evaluated.get());
    }
}
