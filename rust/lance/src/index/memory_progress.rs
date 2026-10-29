// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory observation of index builds, layered over their progress reporting.
//!
//! See `lance_core::memory` for the switch and the metrics.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use lance_core::memory::{MemoryObserver, StageWindow, record_operation_peak, record_stage_peak};
use lance_index::progress::IndexBuildProgress;
use lance_index::scalar::{BuiltinIndexType, LANCE_SCALAR_INDEX, ScalarIndexParams};
use lance_index::{IndexParams, IndexType};

use super::vector::{LANCE_VECTOR_INDEX, VectorIndexParams};
use crate::Result;

/// The operation an observed build belongs to, reported as its `operation` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservedOperation {
    CreateIndex,
    MergeIndexMetadata,
}

impl ObservedOperation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CreateIndex => "create_index",
            Self::MergeIndexMetadata => "merge_index_metadata",
        }
    }
}

/// An index build whose memory is observed while `LANCE_MEMORY_OBSERVE` is on.
///
/// The build reports progress through [`Self::progress`], which records the
/// peak memory of every stage; [`Self::finish`] records the peak of the whole
/// build.
pub struct ObservedBuild {
    progress: Arc<dyn IndexBuildProgress>,
    build: Option<(StageWindow, ObservedOperation, String)>,
}

impl ObservedBuild {
    /// A build that is not observed: `progress` is handed through unchanged.
    pub fn unobserved(progress: Arc<dyn IndexBuildProgress>) -> Self {
        Self {
            progress,
            build: None,
        }
    }

    /// Observes the build with the process-wide observer when
    /// `LANCE_MEMORY_OBSERVE` is on; `index_type` is only evaluated then.
    pub fn start(
        progress: Arc<dyn IndexBuildProgress>,
        operation: ObservedOperation,
        index_type: impl FnOnce() -> String,
    ) -> Result<Self> {
        Ok(match MemoryObserver::global()? {
            Some(observer) => Self::with_observer(progress, observer, operation, index_type()),
            None => Self::unobserved(progress),
        })
    }

    pub fn with_observer(
        progress: Arc<dyn IndexBuildProgress>,
        observer: Arc<MemoryObserver>,
        operation: ObservedOperation,
        index_type: String,
    ) -> Self {
        let build = observer.open_window();
        let progress = Arc::new(MemoryObservingProgress {
            inner: progress,
            observer,
            operation,
            index_type: index_type.clone(),
            stages: Mutex::default(),
        });
        Self {
            progress,
            build: Some((build, operation, index_type)),
        }
    }

    /// The progress reporter to hand to the build.
    pub fn progress(&self) -> Arc<dyn IndexBuildProgress> {
        Arc::clone(&self.progress)
    }

    /// Records the peak memory of the whole build, once it has succeeded. A
    /// build dropped without this records nothing.
    pub fn finish(self) {
        if let Some((window, operation, index_type)) = self.build
            && let Some(peak_bytes) = window.close()
        {
            record_operation_peak(operation.as_str(), &index_type, peak_bytes);
        }
    }
}

/// The `index_type` label of a build, resolved the way the build resolves the
/// index it trains.
pub fn index_type_label(index_type: IndexType, params: &dyn IndexParams) -> String {
    if let Some(vector) = params.as_any().downcast_ref::<VectorIndexParams>() {
        return vector.index_type().to_string().to_lowercase();
    }
    // Only the generic scalar type trains the plugin its parameters name;
    // the specific ones replace the parameters with their own.
    if index_type == IndexType::Scalar
        && let Some(scalar) = params.as_any().downcast_ref::<ScalarIndexParams>()
    {
        let plugin = scalar.index_type.to_lowercase();
        return if plugin == "fts" {
            BuiltinIndexType::Inverted.as_str().to_string()
        } else {
            plugin
        };
    }
    if let Ok(builtin) = BuiltinIndexType::try_from(index_type) {
        return builtin.as_str().to_string();
    }
    let name = params.index_name();
    if name != LANCE_SCALAR_INDEX && name != LANCE_VECTOR_INDEX {
        // An index extension.
        return name.to_lowercase();
    }
    index_type.to_string().to_lowercase()
}

/// Forwards every progress call to `inner` and records the peak memory of
/// each stage between its start and its completion.
#[derive(Debug)]
struct MemoryObservingProgress {
    inner: Arc<dyn IndexBuildProgress>,
    observer: Arc<MemoryObserver>,
    operation: ObservedOperation,
    index_type: String,
    /// Open windows by stage name. A stage may run nested in, or again after,
    /// another stage of the same name, so each name keeps a stack.
    stages: Mutex<HashMap<String, Vec<StageWindow>>>,
}

impl MemoryObservingProgress {
    fn stages(&self) -> MutexGuard<'_, HashMap<String, Vec<StageWindow>>> {
        self.stages.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl IndexBuildProgress for MemoryObservingProgress {
    async fn stage_start(&self, stage: &str, total: Option<u64>, unit: &str) -> Result<()> {
        let window = self.observer.open_window();
        self.stages()
            .entry(stage.to_string())
            .or_default()
            .push(window);
        self.inner.stage_start(stage, total, unit).await
    }

    async fn stage_progress(&self, stage: &str, completed: u64) -> Result<()> {
        self.inner.stage_progress(stage, completed).await
    }

    async fn stage_complete(&self, stage: &str) -> Result<()> {
        let window = self.stages().get_mut(stage).and_then(Vec::pop);
        if let Some(peak_bytes) = window.and_then(StageWindow::close) {
            record_stage_peak(self.operation.as_str(), &self.index_type, stage, peak_bytes);
        }
        self.inner.stage_complete(stage).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use lance_linalg::distance::MetricType;
    use rstest::rstest;
    use std::any::Any;
    use std::time::Duration;

    lance_testing::define_stage_event_progress!(RecordingProgress, IndexBuildProgress, Result<()>);

    #[derive(Debug)]
    struct FailingProgress;

    #[async_trait]
    impl IndexBuildProgress for FailingProgress {
        async fn stage_start(&self, stage: &str, _: Option<u64>, _: &str) -> Result<()> {
            Err(Error::invalid_input(format!("start {stage}")))
        }
        async fn stage_progress(&self, stage: &str, _: u64) -> Result<()> {
            Err(Error::invalid_input(format!("progress {stage}")))
        }
        async fn stage_complete(&self, stage: &str) -> Result<()> {
            Err(Error::invalid_input(format!("complete {stage}")))
        }
    }

    struct ExtensionParams;

    impl IndexParams for ExtensionParams {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn index_name(&self) -> &str {
            "My_Extension"
        }
    }

    fn observed(inner: Arc<dyn IndexBuildProgress>) -> ObservedBuild {
        ObservedBuild::with_observer(
            inner,
            Arc::new(MemoryObserver::new(Duration::from_millis(1))),
            ObservedOperation::CreateIndex,
            "ivf_pq".to_string(),
        )
    }

    #[tokio::test]
    async fn test_every_call_reaches_the_inner_progress_in_order() {
        let inner = Arc::new(RecordingProgress::default());
        let build = observed(inner.clone());
        let progress = build.progress();
        progress
            .stage_start("train_ivf", Some(3), "iterations")
            .await
            .unwrap();
        progress.stage_progress("train_ivf", 1).await.unwrap();
        progress.stage_start("nested", None, "").await.unwrap();
        progress.stage_complete("nested").await.unwrap();
        progress.stage_progress("train_ivf", 3).await.unwrap();
        progress.stage_complete("train_ivf").await.unwrap();
        progress.stage_complete("never_started").await.unwrap();
        progress.stage_progress("never_started", 7).await.unwrap();
        build.finish();

        let expected = [
            ("start", "train_ivf", 3),
            ("progress", "train_ivf", 1),
            ("start", "nested", 0),
            ("complete", "nested", 0),
            ("progress", "train_ivf", 3),
            ("complete", "train_ivf", 0),
            ("complete", "never_started", 0),
            ("progress", "never_started", 7),
        ]
        .map(|(event, stage, value)| (event.to_string(), stage.to_string(), value));
        assert_eq!(inner.recorded_events(), expected);
    }

    #[tokio::test]
    async fn test_inner_errors_come_back_unchanged() {
        let progress = observed(Arc::new(FailingProgress)).progress();
        let start = progress.stage_start("load_data", None, "rows").await;
        let update = progress.stage_progress("load_data", 1).await;
        let complete = progress.stage_complete("load_data").await;
        for (result, message) in [
            (start, "start load_data"),
            (update, "progress load_data"),
            (complete, "complete load_data"),
        ] {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    #[test]
    fn test_unobserved_build_hands_the_progress_through() {
        let inner: Arc<dyn IndexBuildProgress> = Arc::new(RecordingProgress::default());
        let build = ObservedBuild::unobserved(inner.clone());
        assert!(Arc::ptr_eq(&build.progress(), &inner));
    }

    #[rstest]
    #[case::scalar_plugin(IndexType::Scalar, Box::new(ScalarIndexParams::new("BTree".to_string())), "btree")]
    #[case::scalar_fts(IndexType::Scalar, Box::new(ScalarIndexParams::new("fts".to_string())), "inverted")]
    #[case::builtin_ignores_default_params(
        IndexType::Bitmap,
        Box::new(ScalarIndexParams::default()),
        "bitmap"
    )]
    #[case::vector(
        IndexType::Vector,
        Box::new(VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 10)),
        "ivf_pq"
    )]
    #[case::vector_type_from_params(
        IndexType::IvfPq,
        Box::new(VectorIndexParams::ivf_flat(4, MetricType::L2)),
        "ivf_flat"
    )]
    #[case::extension(IndexType::Vector, Box::new(ExtensionParams), "my_extension")]
    fn test_index_type_label(
        #[case] index_type: IndexType,
        #[case] params: Box<dyn IndexParams>,
        #[case] expected: &str,
    ) {
        assert_eq!(index_type_label(index_type, params.as_ref()), expected);
    }

    #[cfg(all(feature = "metrics", target_os = "linux"))]
    mod peaks {
        use super::*;
        use lance_core::memory::{METRIC_OPERATION_PEAK, METRIC_STAGE_PEAK};
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        /// A recorded peak as `(metric name, labels, value)`.
        type RecordedPeak = (String, Vec<(String, String)>, f64);

        /// Peaks recorded by `run`.
        fn record_peaks(run: impl AsyncFnOnce()) -> Vec<RecordedPeak> {
            let recorder = DebuggingRecorder::new();
            let snapshotter = recorder.snapshotter();
            ::metrics::with_local_recorder(&recorder, || {
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap()
                    .block_on(run());
            });
            snapshotter
                .snapshot()
                .into_vec()
                .into_iter()
                .flat_map(|(key, _unit, _description, value)| {
                    let key = key.key().clone();
                    let labels: Vec<(String, String)> = key
                        .labels()
                        .map(|label| (label.key().to_string(), label.value().to_string()))
                        .collect();
                    let values = match value {
                        DebugValue::Histogram(values) => values,
                        _ => Vec::new(),
                    };
                    values
                        .into_iter()
                        .map(move |value| (key.name().to_string(), labels.clone(), value.0))
                })
                .collect()
        }

        fn stage_peaks(recorded: &[RecordedPeak], stage: &str) -> usize {
            recorded
                .iter()
                .filter(|(name, labels, value)| {
                    name == METRIC_STAGE_PEAK
                        && labels.iter().any(|(k, v)| k == "stage" && v == stage)
                        && *value > 0.0
                })
                .count()
        }

        #[test]
        fn test_every_completed_stage_and_the_build_record_a_peak() {
            let recorded = record_peaks(async || {
                let build = observed(Arc::new(RecordingProgress::default()));
                let progress = build.progress();
                for stage in ["write_root_index", "rebuild_hnsw_graph"] {
                    progress.stage_start(stage, None, "").await.unwrap();
                }
                for stage in ["rebuild_hnsw_graph", "write_root_index"] {
                    progress.stage_complete(stage).await.unwrap();
                }
                for _ in 0..2 {
                    progress
                        .stage_start("load_data", None, "rows")
                        .await
                        .unwrap();
                    progress.stage_complete("load_data").await.unwrap();
                }
                build.finish();
            });
            assert_eq!(stage_peaks(&recorded, "write_root_index"), 1);
            assert_eq!(stage_peaks(&recorded, "rebuild_hnsw_graph"), 1);
            assert_eq!(stage_peaks(&recorded, "load_data"), 2);
            let labels = |name: &str| {
                recorded
                    .iter()
                    .filter(|(metric, _, _)| metric == name)
                    .map(|(_, labels, _)| labels.clone())
                    .next()
            };
            assert_eq!(
                labels(METRIC_OPERATION_PEAK),
                Some(vec![
                    ("operation".to_string(), "create_index".to_string()),
                    ("index_type".to_string(), "ivf_pq".to_string()),
                ])
            );
        }

        #[test]
        fn test_unfinished_work_records_nothing() {
            let recorded = record_peaks(async || {
                let build = observed(Arc::new(RecordingProgress::default()));
                build
                    .progress()
                    .stage_start("shuffle", None, "rows")
                    .await
                    .unwrap();
                // The build fails here: the stage never completes and the
                // build is dropped without `finish`.
                drop(build);
            });
            assert!(recorded.is_empty(), "{recorded:?}");
        }
    }
}
