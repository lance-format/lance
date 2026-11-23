// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::time::{Duration, Instant};

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
pub async fn execute_with_retry<E: RetryExecutor>(
    executor: E,
    dataset: Arc<Dataset>,
    config: RetryConfig,
) -> Result<E::Result> {
    let start = Instant::now();
    let mut dataset_ref = dataset;
    let mut backoff = SlotBackoff::default();

    while backoff.attempt() <= config.max_retries {
        let mut executor_clone = executor.clone();
        executor_clone.update_dataset(dataset_ref.clone());

        let execute_fut = executor_clone.execute_impl();
        let execute_fut =
            maybe_timeout(backoff.attempt(), start, config.retry_timeout, execute_fut);
        let data = execute_fut.await??;

        if backoff.attempt() > 0 && start.elapsed() > config.retry_timeout {
            return Err(timeout_error(config.retry_timeout, backoff.attempt() + 1));
        }

        // Cancelling a commit can report failure after its manifest has already
        // been published. Once started, let it finish and return its actual outcome.
        match executor.commit(dataset_ref.clone(), data).await {
            Ok(result) => return Ok(result),
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

    use arrow_array::record_batch;
    use rstest::rstest;

    use super::*;
    use crate::dataset::{DeleteBuilder, InsertBuilder, WriteParams};

    #[derive(Clone)]
    struct DelayedRetryExecutor {
        commit_calls: Arc<AtomicUsize>,
        delay_commit: bool,
    }

    impl RetryExecutor for DelayedRetryExecutor {
        type Data = ();
        type Result = Arc<Dataset>;

        async fn execute_impl(&self) -> Result<Self::Data> {
            if self.commit_calls.load(Ordering::SeqCst) > 0 && !self.delay_commit {
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
            Ok(())
        }

        async fn commit(&self, dataset: Arc<Dataset>, _: Self::Data) -> Result<Self::Result> {
            if self.commit_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(Error::retryable_commit_conflict_source(
                    dataset.version().version,
                    "injected conflict".into(),
                ));
            }

            let result = DeleteBuilder::new(dataset, "true").execute().await?;
            // The manifest is already durable, but the caller is still awaiting
            // the commit response when the retry deadline expires.
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok(result.new_dataset)
        }

        fn update_dataset(&mut self, _: Arc<Dataset>) {}
    }

    #[rstest]
    #[case::during_execution(false)]
    #[case::after_commit(true)]
    #[tokio::test(start_paused = true)]
    async fn test_retry_timeout_respects_commit_boundary(#[case] delay_commit: bool) {
        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                max_rows_per_file: 1,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [1, 2])).unwrap()])
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        let commit_calls = Arc::new(AtomicUsize::new(0));
        let result = execute_with_retry(
            DelayedRetryExecutor {
                commit_calls: commit_calls.clone(),
                delay_commit,
            },
            Arc::new(dataset.clone()),
            RetryConfig {
                max_retries: 1,
                retry_timeout: Duration::from_secs(5),
            },
        )
        .await;

        dataset.checkout_latest().await.unwrap();
        if delay_commit {
            assert_eq!(dataset.version().version, 2);
            assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
            let committed = result.expect("a landed commit must be reported as successful");
            assert_eq!(committed.version().version, dataset.version().version);
            assert_eq!(commit_calls.load(Ordering::SeqCst), 2);
        } else {
            let error = result.expect_err("execution must still respect the retry timeout");
            assert!(matches!(&error, Error::TooMuchWriteContention { .. }));
            assert!(error.to_string().contains("failed on retry_timeout"));
            assert_eq!(dataset.version().version, 1);
            assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
            assert_eq!(commit_calls.load(Ordering::SeqCst), 1);
        }
    }
}
