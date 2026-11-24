// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub(super) use lance_table_runtime::commit::retry::{
    RetryConfig, RetryExecutor, execute_with_retry,
};

impl lance_table_runtime::commit::retry::RetrySnapshot for crate::Dataset {
    async fn checkout_latest(&mut self) -> crate::Result<()> {
        self.checkout_latest().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use arrow_array::record_batch;
    use lance_core::{Error, Result};
    use rstest::rstest;

    use super::*;
    use crate::Dataset;
    use crate::dataset::{DeleteBuilder, InsertBuilder, WriteParams};

    /// Fails its first commit with a retryable conflict. On the retry it
    /// either stalls in execution, or publishes a real delete and then stalls
    /// before returning, both past the retry deadline.
    #[derive(Clone)]
    struct DelayedRetryExecutor {
        commit_calls: Arc<AtomicUsize>,
        delay_commit: bool,
    }

    impl RetryExecutor<Dataset> for DelayedRetryExecutor {
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
            // The manifest is durable, but the deadline expires before the
            // commit returns.
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
            // The landed version is reported as the success it is.
            let committed = result.expect("a landed commit must be reported as successful");
            assert_eq!(dataset.version().version, 2);
            assert_eq!(committed.version().version, 2);
            assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
            assert_eq!(commit_calls.load(Ordering::SeqCst), 2);
        } else {
            // Execution still honours the deadline, and nothing is committed.
            let error = result.expect_err("execution must still respect the retry timeout");
            assert!(matches!(&error, Error::TooMuchWriteContention { .. }));
            assert!(error.to_string().contains("failed on retry_timeout"));
            assert_eq!(dataset.version().version, 1);
            assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
            assert_eq!(commit_calls.load(Ordering::SeqCst), 1);
        }
    }
}
