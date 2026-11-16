// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub(super) use lance_table_runtime::commit::retry::{
    RetryConfig, RetryExecutor, execute_with_retry_and_stages as execute_with_retry,
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
    use lance_core::memory::{MemoryObserver, ObservedOperation, OperationScope};
    use lance_core::utils::tracing::TRACE_MEMORY;
    use lance_core::{Error, Result};
    use rstest::rstest;
    use tracing_mock::{expect, subscriber};

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
            &OperationScope::unobserved().stages(),
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

    /// Has nothing to execute, and loses the race to commit once.
    #[derive(Clone, Default)]
    struct ConflictsOnce(Arc<AtomicUsize>);

    impl RetryExecutor<Dataset> for ConflictsOnce {
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
        let batch = record_batch!(("id", Int32, [1, 2, 3])).unwrap();
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
