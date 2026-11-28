// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{FutureExt, TryFutureExt, future::Either};
use lance_core::memory::{OperationScope, StageReporter};
use lance_core::utils::backoff::SlotBackoff;
use lance_core::{Error, Result};

/// Reload a write operation's snapshot before retrying a conflict.
pub trait RetrySnapshot: Clone {
    /// Advance the snapshot to the latest version on its branch.
    fn checkout_latest(&mut self) -> impl Future<Output = Result<()>> + Send;
}

impl RetrySnapshot for crate::Table {
    async fn checkout_latest(&mut self) -> Result<()> {
        self.checkout_latest().await
    }
}

/// Wall-clock budget for conflict retry backoff when callers do not override it.
/// It starts once the commit has caught up with the versions committed since
/// its read version, so how far behind the writer started does not count.
pub const DEFAULT_COMMIT_RETRY_TIMEOUT: Duration = Duration::from_secs(30);

/// Env var overriding [`DEFAULT_COMMIT_RETRY_TIMEOUT`] process-wide, in
/// (possibly fractional) seconds: the operational escape hatch for callers
/// that have no explicit-timeout API of their own.
const COMMIT_RETRY_TIMEOUT_ENV: &str = "LANCE_COMMIT_RETRY_TIMEOUT_SECS";

/// The commit conflict-retry budget used when the caller does not set one:
/// `LANCE_COMMIT_RETRY_TIMEOUT_SECS` if set to a valid positive number of seconds,
/// otherwise [`DEFAULT_COMMIT_RETRY_TIMEOUT`]. Read once per process.
pub fn default_commit_retry_timeout() -> Duration {
    static TIMEOUT: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *TIMEOUT.get_or_init(|| {
        parse_commit_retry_timeout(std::env::var(COMMIT_RETRY_TIMEOUT_ENV).ok().as_deref())
    })
}

/// Parse `LANCE_COMMIT_RETRY_TIMEOUT_SECS`'s value; `None` means unset. Anything
/// that is not a finite positive number of seconds warns and falls back to
/// [`DEFAULT_COMMIT_RETRY_TIMEOUT`].
fn parse_commit_retry_timeout(raw: Option<&str>) -> Duration {
    let Some(raw) = raw else {
        return DEFAULT_COMMIT_RETRY_TIMEOUT;
    };
    match raw.trim().parse::<f64>() {
        Ok(secs) if secs.is_finite() && secs > 0.0 => Duration::from_secs_f64(secs),
        _ => {
            log::warn!(
                "ignoring invalid {COMMIT_RETRY_TIMEOUT_ENV}={raw:?}; using the {}s default",
                DEFAULT_COMMIT_RETRY_TIMEOUT.as_secs()
            );
            DEFAULT_COMMIT_RETRY_TIMEOUT
        }
    }
}

pub(crate) fn timeout_error(retry_timeout: Duration, attempts: u32) -> Error {
    Error::too_much_write_contention(format!(
        "Attempted {} times, but failed on retry_timeout of {:.3} seconds.",
        attempts,
        retry_timeout.as_secs_f32()
    ))
}

pub(crate) fn maybe_timeout<T>(
    attempt: u32,
    start: Instant,
    retry_timeout: Duration,
    future: impl Future<Output = T>,
) -> impl Future<Output = Result<T>> {
    if attempt == 0 {
        // The first attempt establishes the observed latency used by SlotBackoff.
        Either::Left(future.map(Ok))
    } else {
        let remaining = retry_timeout.saturating_sub(start.elapsed());
        Either::Right(
            tokio::time::timeout(remaining, future)
                .map_err(move |_| timeout_error(retry_timeout, attempt + 1)),
        )
    }
}

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
pub trait RetryExecutor<S: RetrySnapshot>: Clone {
    type Data;
    type Result;

    /// Execute the operation logic without committing
    fn execute_impl(&self) -> impl Future<Output = Result<Self::Data>> + Send;

    /// Commit the operation data
    fn commit(
        &self,
        dataset: Arc<S>,
        data: Self::Data,
    ) -> impl Future<Output = Result<Self::Result>> + Send;

    /// Update the dataset reference for retry attempts
    fn update_dataset(&mut self, dataset: Arc<S>);
}

/// Execute an operation with retry logic for commit conflicts
pub async fn execute_with_retry<S: RetrySnapshot, E: RetryExecutor<S>>(
    executor: E,
    dataset: Arc<S>,
    config: RetryConfig,
) -> Result<E::Result> {
    execute_with_retry_and_stages(
        executor,
        dataset,
        config,
        &OperationScope::unobserved().stages(),
    )
    .await
}

/// Execute an operation and report each attempt's execution and commit to its caller.
#[doc(hidden)]
pub async fn execute_with_retry_and_stages<S: RetrySnapshot, E: RetryExecutor<S>>(
    executor: E,
    dataset: Arc<S>,
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

        // The deadline is checked before the commit starts, never around it.
        // Cancelling a commit after its manifest is written would report a
        // committed write as failed, and callers retrying a non-idempotent
        // operation would apply it twice. The commit is still bounded by its
        // own retry limits.
        if backoff.attempt() > 0 && start.elapsed() > config.retry_timeout {
            return Err(timeout_error(config.retry_timeout, backoff.attempt() + 1));
        }

        let commit_stage = stages.start("commit");
        match executor.commit(dataset_ref.clone(), data).await {
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
    use super::*;

    #[rstest::rstest]
    #[case::unset(None, DEFAULT_COMMIT_RETRY_TIMEOUT)]
    #[case::whole_seconds(Some("600"), Duration::from_secs(600))]
    #[case::fractional_and_padded(Some(" 2.5 "), Duration::from_secs_f64(2.5))]
    #[case::empty(Some(""), DEFAULT_COMMIT_RETRY_TIMEOUT)]
    #[case::not_a_number(Some("abc"), DEFAULT_COMMIT_RETRY_TIMEOUT)]
    #[case::zero(Some("0"), DEFAULT_COMMIT_RETRY_TIMEOUT)]
    #[case::negative(Some("-5"), DEFAULT_COMMIT_RETRY_TIMEOUT)]
    #[case::infinite(Some("inf"), DEFAULT_COMMIT_RETRY_TIMEOUT)]
    #[case::nan(Some("NaN"), DEFAULT_COMMIT_RETRY_TIMEOUT)]
    fn test_parse_commit_retry_timeout(#[case] raw: Option<&str>, #[case] expected: Duration) {
        assert_eq!(parse_commit_retry_timeout(raw), expected);
    }
}
