// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory metrics published through the `metrics` crate facade.
//!
//! Without the `metrics` feature every function here is a no-op.

/// Resident set size of the process, in bytes.
pub const METRIC_RSS: &str = "lance_memory_rss_bytes";
/// Largest resident set size of the process since it started, in bytes.
pub const METRIC_RSS_PEAK: &str = "lance_memory_rss_peak_bytes";
/// Memory charged to the cgroup that sets the tightest limit on the process, or
/// to its own cgroup when no group sets one, in bytes.
pub const METRIC_CGROUP: &str = "lance_memory_cgroup_bytes";
/// Tightest memory limit over the process's cgroup and its ancestors, in bytes.
pub const METRIC_CGROUP_LIMIT: &str = "lance_memory_cgroup_limit_bytes";
/// Peak resident set size during each stage of an index build, in bytes,
/// labelled by `operation`, `index_type` and `stage`.
pub const METRIC_STAGE_PEAK: &str = "lance_memory_stage_peak_bytes";
/// Peak resident set size during a whole index build, in bytes, labelled by
/// `operation` and `index_type`.
pub const METRIC_OPERATION_PEAK: &str = "lance_memory_operation_peak_bytes";

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * MIB;

/// Recommended histogram buckets for the peak metrics, in bytes: powers of two
/// from 16 MiB to 1 TiB.
pub const PEAK_BOUNDS: &[f64] = &[
    16.0 * MIB,
    32.0 * MIB,
    64.0 * MIB,
    128.0 * MIB,
    256.0 * MIB,
    512.0 * MIB,
    GIB,
    2.0 * GIB,
    4.0 * GIB,
    8.0 * GIB,
    16.0 * GIB,
    32.0 * GIB,
    64.0 * GIB,
    128.0 * GIB,
    256.0 * GIB,
    512.0 * GIB,
    1024.0 * GIB,
];

/// Recommended fixed histogram boundaries as `(metric_name, boundaries)`.
pub fn histogram_bounds() -> &'static [(&'static str, &'static [f64])] {
    &[
        (METRIC_STAGE_PEAK, PEAK_BOUNDS),
        (METRIC_OPERATION_PEAK, PEAK_BOUNDS),
    ]
}

/// Register descriptions for every memory metric with the installed recorder.
///
/// Call this after installing a recorder and before enumerating its catalog.
pub fn describe_metrics() {
    #[cfg(feature = "metrics")]
    {
        use metrics::Unit::Bytes;
        metrics::describe_gauge!(METRIC_RSS, Bytes, "Resident set size of the process.");
        metrics::describe_gauge!(
            METRIC_RSS_PEAK,
            Bytes,
            "Largest resident set size of the process since it started."
        );
        metrics::describe_gauge!(
            METRIC_CGROUP,
            Bytes,
            "Memory charged to the cgroup that sets the tightest limit on the process, or to its own cgroup."
        );
        metrics::describe_gauge!(
            METRIC_CGROUP_LIMIT,
            Bytes,
            "Tightest memory limit over the process's cgroup and its ancestors."
        );
        metrics::describe_histogram!(
            METRIC_STAGE_PEAK,
            Bytes,
            "Peak resident set size during each index build stage, by operation, index type and stage."
        );
        metrics::describe_histogram!(
            METRIC_OPERATION_PEAK,
            Bytes,
            "Peak resident set size during each index build, by operation and index type."
        );
    }
}

/// Set the memory gauges to the current readings of the process and its cgroup.
///
/// Does nothing unless [`super::OBSERVE_ENV_VAR`] turns observation on. Call it
/// from the collection loop of the recorder; the Python and Java bridges call
/// it before every snapshot. A gauge whose source is unavailable, such as the
/// cgroup limit of a process without one, is left unset, and a gauge keeps its
/// last value when its source disappears later.
pub fn refresh_metrics() {
    match super::observer::observe_flag() {
        Ok(true) => {
            #[cfg(all(feature = "metrics", target_os = "linux"))]
            publish_gauges(
                super::proc::read_process_memory(),
                super::cgroup::read_cgroup_memory(),
            );
            #[cfg(not(all(feature = "metrics", target_os = "linux")))]
            super::observer::warn_unsupported();
        }
        Ok(false) => {}
        Err(err) => {
            static INVALID: std::sync::Once = std::sync::Once::new();
            INVALID.call_once(|| tracing::warn!("Memory gauges are not refreshed: {err}"));
        }
    }
}

#[cfg(all(feature = "metrics", any(test, target_os = "linux")))]
fn publish_gauges(
    process: Option<super::proc::ProcessMemory>,
    cgroup: Option<super::cgroup::CgroupMemory>,
) {
    if let Some(process) = process {
        metrics::gauge!(METRIC_RSS).set(process.rss_bytes as f64);
        metrics::gauge!(METRIC_RSS_PEAK).set(process.peak_rss_bytes as f64);
    }
    let Some(cgroup) = cgroup else {
        return;
    };
    if let Some(current_bytes) = cgroup.current_bytes {
        metrics::gauge!(METRIC_CGROUP).set(current_bytes as f64);
    }
    if let Some(limit_bytes) = cgroup.limit_bytes {
        metrics::gauge!(METRIC_CGROUP_LIMIT).set(limit_bytes as f64);
    }
}

/// Record the peak resident set size of one stage of an index build.
#[doc(hidden)]
pub fn record_stage_peak(operation: &'static str, index_type: &str, stage: &str, peak_bytes: u64) {
    #[cfg(feature = "metrics")]
    metrics::histogram!(
        METRIC_STAGE_PEAK,
        "operation" => operation,
        "index_type" => labels::INDEX_TYPES.bound(index_type),
        "stage" => labels::STAGES.bound(stage),
    )
    .record(peak_bytes as f64);
    #[cfg(not(feature = "metrics"))]
    let _ = (operation, index_type, stage, peak_bytes);
}

/// Record the peak resident set size of a whole index build.
#[doc(hidden)]
pub fn record_operation_peak(operation: &'static str, index_type: &str, peak_bytes: u64) {
    #[cfg(feature = "metrics")]
    metrics::histogram!(
        METRIC_OPERATION_PEAK,
        "operation" => operation,
        "index_type" => labels::INDEX_TYPES.bound(index_type),
    )
    .record(peak_bytes as f64);
    #[cfg(not(feature = "metrics"))]
    let _ = (operation, index_type, peak_bytes);
}

/// Label values that come from outside Lance, such as the index type and stage
/// names of a scalar index plugin, are bounded so that a plugin cannot grow
/// the number of exported series without limit.
#[cfg(feature = "metrics")]
mod labels {
    use std::sync::{Mutex, Once, PoisonError};

    /// Distinct values one label keeps before new ones are reported as `other`.
    const MAX_VALUES: usize = 64;
    const OVERFLOW: &str = "other";

    pub(super) static INDEX_TYPES: BoundedLabel = BoundedLabel::new("index_type");
    pub(super) static STAGES: BoundedLabel = BoundedLabel::new("stage");

    pub(super) struct BoundedLabel {
        name: &'static str,
        seen: Mutex<Vec<String>>,
        overflowed: Once,
    }

    impl BoundedLabel {
        pub(super) const fn new(name: &'static str) -> Self {
            Self {
                name,
                seen: Mutex::new(Vec::new()),
                overflowed: Once::new(),
            }
        }

        pub(super) fn bound(&self, value: &str) -> String {
            let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
            if seen.iter().any(|known| known == value) {
                return value.to_string();
            }
            if seen.len() < MAX_VALUES {
                seen.push(value.to_string());
                return value.to_string();
            }
            drop(seen);
            self.overflowed.call_once(|| {
                tracing::warn!(
                    "Memory metrics saw more than {MAX_VALUES} distinct `{}` values; \
                     later ones are reported as `{OVERFLOW}`",
                    self.name
                );
            });
            OVERFLOW.to_string()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_new_values_past_the_bound_are_other() {
            let label = BoundedLabel::new("stage");
            for i in 0..MAX_VALUES {
                assert_eq!(label.bound(&format!("stage_{i}")), format!("stage_{i}"));
            }
            assert_eq!(label.bound("one_too_many"), OVERFLOW);
            assert_eq!(label.bound("stage_0"), "stage_0");
        }
    }
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use super::super::cgroup::CgroupMemory;
    use super::super::proc::ProcessMemory;
    use super::*;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use rstest::rstest;

    fn capture(record: impl FnOnce()) -> Vec<(metrics::Key, DebugValue)> {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, record);
        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(key, _unit, _description, value)| (key.key().clone(), value))
            .collect()
    }

    fn gauge(recorded: &[(metrics::Key, DebugValue)], name: &str) -> Option<f64> {
        recorded.iter().find_map(|(key, value)| match value {
            DebugValue::Gauge(value) if key.name() == name => Some(value.0),
            _ => None,
        })
    }

    fn histogram(
        recorded: &[(metrics::Key, DebugValue)],
        name: &str,
        labels: &[(&str, &str)],
    ) -> Vec<f64> {
        recorded
            .iter()
            .find_map(|(key, value)| {
                let key_labels: Vec<(&str, &str)> = key
                    .labels()
                    .map(|label| (label.key(), label.value()))
                    .collect();
                match value {
                    DebugValue::Histogram(values) if key.name() == name && key_labels == labels => {
                        Some(values.iter().map(|value| value.0).collect())
                    }
                    _ => None,
                }
            })
            .unwrap_or_default()
    }

    const PROCESS: ProcessMemory = ProcessMemory {
        rss_bytes: 100,
        peak_rss_bytes: 150,
    };
    const CGROUP: CgroupMemory = CgroupMemory {
        current_bytes: Some(200),
        limit_bytes: Some(1000),
    };

    #[rstest]
    #[case::everything(Some(PROCESS), Some(CGROUP), [Some(100.0), Some(150.0), Some(200.0), Some(1000.0)])]
    #[case::no_cgroup(Some(PROCESS), None, [Some(100.0), Some(150.0), None, None])]
    #[case::no_cgroup_limit(
        Some(PROCESS),
        Some(CgroupMemory { limit_bytes: None, ..CGROUP }),
        [Some(100.0), Some(150.0), Some(200.0), None]
    )]
    #[case::no_process_reading(None, Some(CGROUP), [None, None, Some(200.0), Some(1000.0)])]
    #[case::nothing(None, None, [None, None, None, None])]
    fn test_gauges_report_what_is_available(
        #[case] process: Option<ProcessMemory>,
        #[case] cgroup: Option<CgroupMemory>,
        #[case] expected: [Option<f64>; 4],
    ) {
        let recorded = capture(|| publish_gauges(process, cgroup));
        let gauges = [
            METRIC_RSS,
            METRIC_RSS_PEAK,
            METRIC_CGROUP,
            METRIC_CGROUP_LIMIT,
        ];
        assert_eq!(gauges.map(|name| gauge(&recorded, name)), expected);
    }

    #[test]
    fn test_refresh_reports_nothing_while_observation_is_off() {
        assert!(
            std::env::var_os(super::super::OBSERVE_ENV_VAR).is_none(),
            "this test needs the switch unset"
        );
        let recorded = capture(refresh_metrics);
        assert!(recorded.is_empty(), "{recorded:?}");
    }

    #[test]
    fn test_peaks_are_recorded_with_their_labels() {
        let recorded = capture(|| {
            record_stage_peak("create_index", "btree", "load_data", 1 << 20);
            record_operation_peak("create_index", "btree", 2 << 20);
        });
        assert_eq!(
            histogram(
                &recorded,
                METRIC_STAGE_PEAK,
                &[
                    ("operation", "create_index"),
                    ("index_type", "btree"),
                    ("stage", "load_data"),
                ],
            ),
            vec![f64::from(1 << 20)]
        );
        assert_eq!(
            histogram(
                &recorded,
                METRIC_OPERATION_PEAK,
                &[("operation", "create_index"), ("index_type", "btree")],
            ),
            vec![f64::from(2 << 20)]
        );
    }
}
