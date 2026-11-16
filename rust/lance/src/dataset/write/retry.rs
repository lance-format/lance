// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::time::{Duration, Instant};

use lance_core::memory::StageReporter;
use lance_core::utils::backoff::SlotBackoff;
use lance_core::{Error, Result};

use crate::Dataset;
use crate::io::commit::{maybe_timeout, timeout_error};

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub retry_timeout: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 10,
            retry_timeout: Duration::from_secs(30),
        }
    }
}

/// Trait for operations that can be retried on commit conflicts
pub trait RetryExecutor: Clone {
    type Data;
    type Result;

    /// Execute the operation logic without committing
    async fn execute_impl(&self) -> Result<Self::Data>;

    /// Commit the operation data
    async fn commit(&self, dataset: Arc<Dataset>, data: Self::Data) -> Result<Self::Result>;

    /// Update the dataset reference for retry attempts
    fn update_dataset(&mut self, dataset: Arc<Dataset>);
}

/// Execute an operation with retry logic for commit conflicts
///
/// Every attempt reports its `execute` and `commit` stages to `stages`; a
/// commit that conflicts never completes its stage.
pub async fn execute_with_retry<E: RetryExecutor>(
    executor: E,
    dataset: Arc<Dataset>,
    config: RetryConfig,
    stages: &StageReporter,
) -> Result<E::Result> {
    let start = Instant::now();
    let mut dataset_ref = dataset;
    let mut backoff = SlotBackoff::default();

    while backoff.attempt() <= config.max_retries {
        let mut executor_clone = executor.clone();
        executor_clone.update_dataset(dataset_ref.clone());

        let execute_stage = stages.start("execute");
        let execute_fut = executor_clone.execute_impl();
        let execute_fut =
            maybe_timeout(backoff.attempt(), start, config.retry_timeout, execute_fut);
        let data = execute_fut.await??;
        execute_stage.complete();

        let commit_stage = stages.start("commit");
        let commit_future = executor.commit(dataset_ref.clone(), data);
        let commit_future = maybe_timeout(
            backoff.attempt(),
            start,
            config.retry_timeout,
            commit_future,
        );

        match commit_future.await? {
            Ok(result) => {
                commit_stage.complete();
                return Ok(result);
            }
            Err(Error::RetryableCommitConflict { .. }) => {
                // Check whether we have exhausted our retries *before* we sleep.
                if backoff.attempt() >= config.max_retries {
                    break;
                }
                if start.elapsed() > config.retry_timeout {
                    return Err(timeout_error(config.retry_timeout, backoff.attempt() + 1));
                }
                if backoff.attempt() == 0 {
                    // We add 10% buffer here, to allow concurrent writes to complete.
                    // We pass the first attempt's time to the backoff so it's used
                    // as the unit for backoff time slots.
                    // See SlotBackoff implementation for more details on how this works.
                    backoff = backoff.with_unit((start.elapsed().as_millis() * 11 / 10) as u32);
                }

                let sleep_fut = tokio::time::sleep(backoff.next_backoff());
                let sleep_fut =
                    maybe_timeout(backoff.attempt(), start, config.retry_timeout, sleep_fut);
                sleep_fut.await?;

                let mut ds = dataset_ref.as_ref().clone();
                ds.checkout_latest().await?;
                dataset_ref = Arc::new(ds);
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    Err(Error::too_much_write_contention(format!(
        "Attempted {} retries.",
        config.max_retries
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use lance_core::memory::{MemoryObserver, ObservedOperation, OperationScope};
    use lance_core::utils::tracing::TRACE_MEMORY;
    use tracing_mock::{expect, subscriber};

    use super::*;
    use crate::dataset::InsertBuilder;

    /// Has nothing to execute, and loses the race to commit once.
    #[derive(Clone, Default)]
    struct ConflictsOnce(Arc<AtomicUsize>);

    impl RetryExecutor for ConflictsOnce {
        type Data = ();
        type Result = usize;

        async fn execute_impl(&self) -> Result<()> {
            Ok(())
        }

        async fn commit(&self, dataset: Arc<Dataset>, _: ()) -> Result<usize> {
            match self.0.fetch_add(1, Ordering::SeqCst) {
                0 => Err(Error::retryable_commit_conflict_source(
                    dataset.version().version,
                    "a concurrent commit".into(),
                )),
                commits => Ok(commits),
            }
        }

        fn update_dataset(&mut self, _: Arc<Dataset>) {}
    }

    #[rstest::rstest]
    #[case::retried(1)]
    #[case::out_of_retries(0)]
    #[tokio::test]
    #[serial_test::serial(memory_events)]
    async fn test_every_attempt_reports_its_stages_into_the_operation(#[case] max_retries: u32) {
        let ids: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let batch = RecordBatch::try_from_iter([("id", ids)]).unwrap();
        let dataset = InsertBuilder::new("memory://")
            .execute(vec![batch])
            .await
            .unwrap();

        // The commit that conflicts never completes, whether a retry follows
        // or the retries have run out.
        let mut expected = vec![
            ("operation_start", None),
            ("stage_start", Some("execute")),
            ("stage_complete", Some("execute")),
            ("stage_start", Some("commit")),
        ];
        if max_retries > 0 {
            expected.extend([
                ("stage_start", Some("execute")),
                ("stage_complete", Some("execute")),
                ("stage_start", Some("commit")),
                ("stage_complete", Some("commit")),
                ("operation_complete", None),
            ]);
        }
        let mut mock = subscriber::mock().with_filter(|meta| meta.target() == TRACE_MEMORY);
        for (event, stage) in expected {
            let mut fields = expect::field("event")
                .with_value(&event)
                .and(expect::field("operation").with_value(&"update"));
            if let Some(stage) = stage {
                fields = fields.and(expect::field("stage").with_value(&stage));
            }
            mock = mock.event(expect::event().with_fields(fields));
        }
        let (subscriber, handle) = mock.only().run_with_handle();
        let guard = tracing::subscriber::set_default(subscriber);
        let scope = OperationScope::with_observer(
            Arc::new(MemoryObserver::new(Duration::from_millis(1))),
            ObservedOperation::Update,
            None,
        );
        let config = RetryConfig {
            max_retries,
            ..Default::default()
        };
        match execute_with_retry(
            ConflictsOnce::default(),
            Arc::new(dataset),
            config,
            &scope.stages(),
        )
        .await
        {
            Ok(commits) => {
                assert_eq!((max_retries, commits), (1, 1));
                scope.finish();
            }
            Err(error) => assert!(
                max_retries == 0 && matches!(error, Error::TooMuchWriteContention { .. }),
                "{error:?}"
            ),
        }
        drop(guard);
        handle.assert_finished();
    }
}
