// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reports of observed operations, fed to the metrics and to the memory events.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{events, telemetry};

/// Ids start at a random value in every process, so that a process restarted
/// with the same pid, or another copy of Lance in the same process, does not
/// repeat the ids of the first.
static NEXT_OPERATION_ID: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(rand::random()));

/// Reports an observed operation and its stages to the metrics and as memory
/// events. Every event of the operation carries its id, which tells apart
/// operations that run at the same time.
#[derive(Debug)]
pub struct OperationReport {
    id: u64,
    operation: &'static str,
    index_type: Option<String>,
}

impl OperationReport {
    /// A report for an index operation. It reports nothing until [`Self::start`].
    /// Use [`super::OperationScope`] to observe an operation and its stages.
    #[deprecated(note = "Use OperationScope to observe an operation and its stages")]
    pub fn new(operation: &'static str, index_type: String) -> Self {
        Self::for_operation(operation, Some(index_type))
    }

    /// A report with an id that no other operation of the process has. It
    /// reports nothing until [`Self::start`]. `index_type` names the type of
    /// the index the operation builds; without one, events leave the field out
    /// and peaks are recorded without the label.
    pub(super) fn for_operation(operation: &'static str, index_type: Option<String>) -> Self {
        Self {
            id: NEXT_OPERATION_ID.fetch_add(1, Ordering::Relaxed),
            operation,
            index_type,
        }
    }

    /// Reports that the operation started.
    pub fn start(&self) {
        events::emit(
            "operation_start",
            self.id,
            self.operation,
            self.index_type.as_deref(),
            None,
            None,
        );
    }

    /// Reports that a stage of the operation started.
    pub fn stage_start(&self, stage: &str) {
        events::emit(
            "stage_start",
            self.id,
            self.operation,
            self.index_type.as_deref(),
            Some(stage),
            None,
        );
    }

    /// Reports that a stage of the operation completed, with its peak RSS in
    /// bytes where the platform reports memory.
    pub fn stage_complete(&self, stage: &str, peak_bytes: Option<u64>) {
        if let Some(peak_bytes) = peak_bytes {
            telemetry::record_stage_peak(
                self.operation,
                self.index_type.as_deref(),
                stage,
                peak_bytes,
            );
        }
        events::emit(
            "stage_complete",
            self.id,
            self.operation,
            self.index_type.as_deref(),
            Some(stage),
            peak_bytes,
        );
    }

    /// Reports that the operation completed, with its peak RSS in bytes where
    /// the platform reports memory.
    pub fn complete(&self, peak_bytes: Option<u64>) {
        if let Some(peak_bytes) = peak_bytes {
            telemetry::record_operation_peak(
                self.operation,
                self.index_type.as_deref(),
                peak_bytes,
            );
        }
        events::emit(
            "operation_complete",
            self.id,
            self.operation,
            self.index_type.as_deref(),
            None,
            peak_bytes,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::PoisonError;

    use super::*;
    use crate::memory::EMITTING;
    #[cfg(feature = "metrics")]
    use crate::memory::{METRIC_OPERATION_PEAK, METRIC_STAGE_PEAK};
    use crate::utils::tracing::TRACE_MEMORY;
    #[cfg(feature = "metrics")]
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use tracing_mock::field::ExpectedFields;
    use tracing_mock::{expect, subscriber};

    /// Exactly the fields of a memory event of `report`. The memory readings
    /// are expected where the platform reports them, and by value only for the
    /// cgroup limit, the one reading that stays the same while the test runs.
    fn event_fields(
        report: &OperationReport,
        event: &str,
        stage: Option<&str>,
        peak_bytes: Option<u64>,
    ) -> ExpectedFields {
        let mut fields = expect::field("event")
            .with_value(&event)
            .and(expect::field("pid").with_value(&std::process::id()))
            .and(expect::field("operation_id").with_value(&report.id))
            .and(expect::field("operation").with_value(&report.operation));
        if let Some(index_type) = report.index_type.as_deref() {
            fields = fields.and(expect::field("index_type").with_value(&index_type));
        }
        if let Some(stage) = stage {
            fields = fields.and(expect::field("stage").with_value(&stage));
        }
        if let Some(peak_bytes) = peak_bytes {
            fields = fields.and(expect::field("peak_bytes").with_value(&peak_bytes));
        }
        if super::super::proc::read_process_memory().is_some() {
            fields = fields.and(expect::field("rss_bytes"));
        }
        #[cfg(target_os = "linux")]
        let cgroup = super::super::cgroup::read_cgroup_memory();
        #[cfg(not(target_os = "linux"))]
        let cgroup: Option<super::super::cgroup::CgroupMemory> = None;
        if cgroup.is_some_and(|cgroup| cgroup.current_bytes.is_some()) {
            fields = fields.and(expect::field("cgroup_bytes"));
        }
        if let Some(limit_bytes) = cgroup.and_then(|cgroup| cgroup.limit_bytes) {
            fields = fields.and(expect::field("cgroup_limit_bytes").with_value(&limit_bytes));
        }
        fields.only()
    }

    #[test]
    #[allow(deprecated)]
    fn test_every_report_logs_one_event_with_its_operation_id() {
        let _emitting = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        let build = crate::memory::OperationReport::new("create_index", "ivf_pq".to_string());
        let merge = OperationReport::new("merge_index_metadata", "inverted".to_string());
        let compaction = OperationReport::for_operation("compaction", None);
        assert_ne!(build.id, merge.id);
        assert_ne!(merge.id, compaction.id);
        let mut mock = subscriber::mock();
        for fields in [
            event_fields(&build, "operation_start", None, None),
            event_fields(&merge, "operation_start", None, None),
            event_fields(&build, "stage_start", Some("shuffle"), None),
            event_fields(&merge, "stage_start", Some("remap_partition_files"), None),
            event_fields(&build, "stage_complete", Some("shuffle"), Some(2048)),
            event_fields(
                &merge,
                "stage_complete",
                Some("remap_partition_files"),
                None,
            ),
            event_fields(&merge, "operation_complete", None, None),
            event_fields(&build, "operation_complete", None, Some(4096)),
            event_fields(&compaction, "operation_start", None, None),
            event_fields(&compaction, "stage_complete", Some("commit"), Some(512)),
            event_fields(&compaction, "operation_complete", None, Some(1024)),
        ] {
            mock = mock.event(
                expect::event()
                    .with_target(TRACE_MEMORY)
                    .at_level(tracing::Level::INFO)
                    .with_fields(fields),
            );
        }
        let (subscriber, handle) = mock.only().run_with_handle();
        tracing::subscriber::with_default(subscriber, || {
            build.start();
            merge.start();
            build.stage_start("shuffle");
            merge.stage_start("remap_partition_files");
            build.stage_complete("shuffle", Some(2048));
            merge.stage_complete("remap_partition_files", None);
            merge.complete(None);
            build.complete(Some(4096));
            compaction.start();
            compaction.stage_complete("commit", Some(512));
            compaction.complete(Some(1024));
        });
        handle.assert_finished();
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_only_completions_with_a_peak_are_recorded() {
        let _emitting = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let report = OperationReport::for_operation("create_index", Some("btree".to_string()));
            report.start();
            report.stage_start("load_data");
            report.stage_complete("load_data", None);
            report.stage_complete("load_data", Some(1 << 20));
            report.complete(None);
            report.complete(Some(2 << 20));
            let compaction = OperationReport::for_operation("compaction", None);
            compaction.stage_complete("commit", Some(3 << 20));
            compaction.complete(Some(4 << 20));
        });
        type RecordedMetric = (String, Vec<(String, String)>, Vec<f64>);
        let mut recorded: Vec<RecordedMetric> = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(key, _unit, _description, value)| {
                let labels = key
                    .key()
                    .labels()
                    .map(|label| (label.key().to_string(), label.value().to_string()))
                    .collect();
                let samples = match value {
                    DebugValue::Histogram(values) => values.iter().map(|value| value.0).collect(),
                    _ => Vec::new(),
                };
                (key.key().name().to_string(), labels, samples)
            })
            .collect();
        recorded.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        let labels = |pairs: &[(&str, &str)]| -> Vec<(String, String)> {
            pairs
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect()
        };
        assert_eq!(
            recorded,
            [
                (
                    METRIC_OPERATION_PEAK.to_string(),
                    labels(&[("operation", "compaction")]),
                    vec![f64::from(4 << 20)],
                ),
                (
                    METRIC_OPERATION_PEAK.to_string(),
                    labels(&[("operation", "create_index"), ("index_type", "btree")]),
                    vec![f64::from(2 << 20)],
                ),
                (
                    METRIC_STAGE_PEAK.to_string(),
                    labels(&[("operation", "compaction"), ("stage", "commit")]),
                    vec![f64::from(3 << 20)],
                ),
                (
                    METRIC_STAGE_PEAK.to_string(),
                    labels(&[
                        ("operation", "create_index"),
                        ("index_type", "btree"),
                        ("stage", "load_data"),
                    ]),
                    vec![f64::from(1 << 20)],
                ),
            ]
        );
    }
}
