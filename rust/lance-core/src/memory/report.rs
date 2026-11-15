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
#[doc(hidden)]
#[derive(Debug)]
pub struct OperationReport {
    id: u64,
    operation: &'static str,
    index_type: String,
}

impl OperationReport {
    /// A report with an id that no other operation of the process has. It
    /// reports nothing until [`Self::start`].
    pub fn new(operation: &'static str, index_type: String) -> Self {
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
            &self.index_type,
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
            &self.index_type,
            Some(stage),
            None,
        );
    }

    /// Reports that a stage of the operation completed, with its peak RSS in
    /// bytes where the platform reports memory.
    pub fn stage_complete(&self, stage: &str, peak_bytes: Option<u64>) {
        if let Some(peak_bytes) = peak_bytes {
            telemetry::record_stage_peak(self.operation, &self.index_type, stage, peak_bytes);
        }
        events::emit(
            "stage_complete",
            self.id,
            self.operation,
            &self.index_type,
            Some(stage),
            peak_bytes,
        );
    }

    /// Reports that the operation completed, with its peak RSS in bytes where
    /// the platform reports memory.
    pub fn complete(&self, peak_bytes: Option<u64>) {
        if let Some(peak_bytes) = peak_bytes {
            telemetry::record_operation_peak(self.operation, &self.index_type, peak_bytes);
        }
        events::emit(
            "operation_complete",
            self.id,
            self.operation,
            &self.index_type,
            None,
            peak_bytes,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, PoisonError};

    use super::*;
    #[cfg(feature = "metrics")]
    use crate::memory::{METRIC_OPERATION_PEAK, METRIC_STAGE_PEAK};
    use crate::utils::tracing::TRACE_MEMORY;
    #[cfg(feature = "metrics")]
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use tracing_mock::field::ExpectedFields;
    use tracing_mock::{expect, subscriber};

    /// Taken by every test that emits memory events. Tracing caches a
    /// callsite's interest when it is first hit, and while only one scoped
    /// subscriber exists it asks just the thread that hits it: a test emitting
    /// without a subscriber could then turn the callsite off for the mock of a
    /// test running in parallel.
    static EMITTING: Mutex<()> = Mutex::new(());

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
            .and(expect::field("operation").with_value(&report.operation))
            .and(expect::field("index_type").with_value(&report.index_type.as_str()));
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
    fn test_every_report_logs_one_event_with_its_operation_id() {
        let _emitting = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        let build = OperationReport::new("create_index", "ivf_pq".to_string());
        let merge = OperationReport::new("merge_index_metadata", "inverted".to_string());
        assert_ne!(build.id, merge.id);
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
            let report = OperationReport::new("create_index", "btree".to_string());
            report.start();
            report.stage_start("load_data");
            report.stage_complete("load_data", None);
            report.stage_complete("load_data", Some(1 << 20));
            report.complete(None);
            report.complete(Some(2 << 20));
        });
        let mut recorded: Vec<(String, Vec<f64>)> = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(key, _unit, _description, value)| {
                let samples = match value {
                    DebugValue::Histogram(values) => values.iter().map(|value| value.0).collect(),
                    _ => Vec::new(),
                };
                (key.key().name().to_string(), samples)
            })
            .collect();
        recorded.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            recorded,
            [
                (METRIC_OPERATION_PEAK.to_string(), vec![f64::from(2 << 20)]),
                (METRIC_STAGE_PEAK.to_string(), vec![f64::from(1 << 20)]),
            ]
        );
    }
}
