// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-process consumer cancellation without abandoning the other complete drains.

use std::{future::pending, sync::Arc, time::Instant};

use lance::Dataset;
use lance_core::utils::allocation::AllocationBudget;
use serde::Serialize;
use tokio::sync::oneshot;

use super::{DrainConfig, Output, Result, Sink, USAGE, drain, verify};

fn validate(requests: usize, after_batches: u64, expected_rows: u64) -> Result<()> {
    if requests < 2 {
        return Err("run-cancel requires at least two requests".into());
    }
    if after_batches == 0 || after_batches >= expected_rows {
        return Err("CANCEL_AFTER_BATCHES must be positive and less than EXPECTED_ROWS".into());
    }
    Ok(())
}

pub fn parse(args: &mut Vec<String>) -> Result<Option<u64>> {
    if args.first().is_none_or(|arg| arg != "run-cancel") {
        return Ok(None);
    }
    if !matches!(args.len(), 11 | 12) {
        return Err(USAGE.into());
    }
    let after_batches = args[10].parse()?;
    validate(args[8].parse()?, after_batches, args[3].parse()?)?;
    args.remove(10);
    args[0] = "run-many".into();
    Ok(Some(after_batches))
}

#[derive(Debug, Serialize)]
pub struct Cancelled {
    request_index: usize,
    after_batches: u64,
    // Deliberately separate from the complete outputs and without a completion time.
    prefix: Output,
    observed_seconds: Option<f64>,
}

pub struct Trigger {
    after_batches: u64,
    ready: Option<oneshot::Sender<Cancelled>>,
}

impl Trigger {
    pub async fn after_batch(&mut self, sink: &Sink) -> Result<()> {
        if sink.batches != self.after_batches {
            return Ok(());
        }
        self.ready
            .take()
            .ok_or("cancellation trigger already consumed")?
            .send(Cancelled {
                request_index: 0,
                after_batches: self.after_batches,
                prefix: sink.snapshot(),
                observed_seconds: sink.started.map(|t| t.elapsed().as_secs_f64()),
            })
            .map_err(|_| "cancellation observer dropped")?;
        // The enclosing drain still owns its batch and stream here. Only dropping
        // that actual future releases the consumer, its alias and prefetch producer.
        pending::<Result<()>>().await
    }
}

pub async fn run(
    dataset: &Dataset,
    config: &DrainConfig,
    budget: &Arc<AllocationBudget>,
    expected_rows: u64,
    expected_hash: &str,
    after_batches: u64,
    started: Option<Instant>,
) -> Result<(Cancelled, Vec<Output>)> {
    validate(config.requests, after_batches, expected_rows)?;
    if config.pool_bytes == 0 {
        return Err("shared pool bytes must be positive".into());
    }
    let result = {
        let (ready, observed) = oneshot::channel();
        let cancel_first = async {
            let mut first = Box::pin(drain(
                dataset,
                config,
                budget,
                started,
                Some(Trigger {
                    after_batches,
                    ready: Some(ready),
                }),
            ));
            let cancelled: Result<Cancelled> = tokio::select! {
                // Preserve a real reader error instead of reporting its closed
                // notification channel when both branches become ready together.
                biased;
                output = &mut first => {
                    output?;
                    Err("request completed before cancellation point".into())
                }
                cancelled = observed => Ok(cancelled?),
            };
            drop(first);
            let cancelled = cancelled?;
            if cancelled.prefix.rows >= expected_rows {
                return Err("cancellation must precede complete output".into());
            }
            Ok(cancelled)
        };
        let survivors = futures::future::try_join_all((1..config.requests).map(|_| async {
            let output = drain(dataset, config, budget, started, None).await?;
            verify(&output, expected_rows, expected_hash)?;
            Ok(output)
        }));
        futures::try_join!(cancel_first, survivors)
    };
    // All consumer futures have completed or dropped before waiting for detached
    // I/O/CPU owners. The cancelled prefix contains only counters and a digest.
    budget.wait_for_capacity(config.pool_bytes).await?;
    if budget.reserved() != 0 {
        return Err("cancelled request group retained allocation credit".into());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConsumerExecutor, Fixture, Mode, drain_all, fixture_batch, prepare, run_command};
    use rstest::rstest;
    use std::time::Duration;

    #[rstest]
    #[case::defaults(None)]
    #[case::explicit_attempt(Some("16384"))]
    fn cancellation_controls_preserve_the_drain_configuration(#[case] attempt: Option<&str>) {
        let mut args = [
            "run-cancel",
            "prefetched",
            "uri",
            "17",
            "hash",
            "2",
            "65536",
            "10",
            "3",
            "65536",
            "4",
        ]
        .map(String::from)
        .to_vec();
        if let Some(attempt) = attempt {
            args.push(attempt.into());
        }
        assert_eq!(parse(&mut args).unwrap(), Some(4));
        assert_eq!(args[0], "run-many");
        let config = DrainConfig::from_run_args(&args).unwrap();
        assert_eq!(config.requests, 3);
        assert_eq!(config.pool_bytes, 65536);
        assert_eq!(config.workspace, 65536);
        assert_eq!(config.batch_rows, 2);
        assert_eq!(config.slow, Duration::from_micros(10));
        assert_eq!(
            config.attempt_workspace,
            attempt.map(|s| s.parse().unwrap())
        );
    }

    #[rstest]
    #[case::zero("0", "3", "must be positive")]
    #[case::past_rows("17", "3", "less than EXPECTED_ROWS")]
    #[case::no_survivor("1", "1", "at least two requests")]
    fn invalid_cancellation_controls_fail_before_open(
        #[case] batches: &str,
        #[case] requests: &str,
        #[case] message: &str,
    ) {
        let mut args = [
            "run-cancel",
            "admitted",
            "unused",
            "17",
            "hash",
            "2",
            "65536",
            "0",
            requests,
            "65536",
            batches,
        ]
        .map(String::from)
        .to_vec();
        assert!(parse(&mut args).unwrap_err().to_string().contains(message));
    }

    #[rstest]
    #[tokio::test]
    async fn cancellation_point_still_owns_the_live_output(
        #[values(Mode::Admitted, Mode::Prefetched)] mode: Mode,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("data.lance").to_str().unwrap().to_string();
        prepare(Fixture::FullZip, &uri, 17, 67, 5).await.unwrap();
        let dataset = Dataset::open(&uri).await.unwrap();
        let config = DrainConfig {
            mode,
            batch_rows: 2,
            workspace: 65536,
            attempt_workspace: None,
            slow: Duration::ZERO,
            requests: 3,
            pool_bytes: 65536,
        };
        let budget = AllocationBudget::new(config.pool_bytes);
        let (ready, observed) = oneshot::channel();
        let mut first = Box::pin(drain(
            &dataset,
            &config,
            &budget,
            None,
            Some(Trigger {
                after_batches: 1,
                ready: Some(ready),
            }),
        ));
        let prefix = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::select! {
                output = &mut first => panic!("consumer returned before cancellation: {output:?}"),
                prefix = observed => prefix.unwrap(),
            }
        })
        .await
        .unwrap();
        assert_eq!(prefix.prefix.batches, 1);
        assert!(budget.reserved() > 0);
        assert!(budget.try_reserve(config.pool_bytes).is_none());
        assert!(futures::poll!(&mut first).is_pending());
        drop(first);
        tokio::time::timeout(
            Duration::from_secs(5),
            budget.wait_for_capacity(config.pool_bytes),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(budget.reserved(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn cancelling_one_retained_batch_preserves_survivors_and_pool_reuse(
        #[values(Mode::Ordinary, Mode::Admitted, Mode::Prefetched)] mode: Mode,
        #[values(1, 4)] after_batches: u64,
        #[values(ConsumerExecutor::Caller, ConsumerExecutor::Worker)] executor: ConsumerExecutor,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("data.lance").to_str().unwrap().to_string();
        let expected = prepare(Fixture::FullZip, &uri, 17, 67, 5).await.unwrap();
        let dataset = Dataset::open(&uri).await.unwrap();
        let config = DrainConfig {
            mode,
            batch_rows: 2,
            workspace: 65536,
            attempt_workspace: None,
            slow: Duration::ZERO,
            requests: 3,
            pool_bytes: 65536,
        };
        let budget = AllocationBudget::new(config.pool_bytes);
        let task_dataset = dataset.clone();
        let task_config = config.clone();
        let task_budget = budget.clone();
        let hash = expected.hash.clone();
        let (cancelled, survivors) = tokio::time::timeout(
            Duration::from_secs(5),
            executor.run(async move {
                run(
                    &task_dataset,
                    &task_config,
                    &task_budget,
                    17,
                    &hash,
                    after_batches,
                    None,
                )
                .await
            }),
        )
        .await
        .expect("cancellation must not strand other consumers")
        .unwrap();
        assert_eq!(cancelled.request_index, 0);
        assert_eq!(cancelled.after_batches, after_batches);
        assert_eq!(cancelled.prefix.batches, after_batches);
        assert!(cancelled.prefix.rows > 0 && cancelled.prefix.rows < expected.rows);
        assert_eq!(cancelled.prefix.completed_seconds, None);
        let mut oracle = Sink::new(None);
        oracle
            .consume(
                &fixture_batch(Fixture::FullZip, 0, cancelled.prefix.rows as usize, 67).unwrap(),
            )
            .unwrap();
        let prefix = oracle.finish();
        assert_eq!(cancelled.prefix.hash, prefix.hash);
        assert_eq!(cancelled.prefix.payload_bytes, prefix.payload_bytes);
        assert_eq!(survivors.len(), 2);
        for output in survivors {
            verify(&output, expected.rows, &expected.hash).unwrap();
            assert_eq!(output.payload_bytes, expected.payload_bytes);
        }
        assert_eq!(budget.reserved(), 0);
        let retry = tokio::time::timeout(
            Duration::from_secs(5),
            drain_all(
                &dataset,
                &config,
                &budget,
                expected.rows,
                &expected.hash,
                None,
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(retry.len(), 3);
        assert_eq!(budget.reserved(), 0);
    }

    #[rstest]
    #[case::wrong_survivor_hash(true, 1, "incomplete or altered output")]
    #[case::unreachable_point(false, 16, "request completed before cancellation point")]
    #[tokio::test]
    async fn failed_or_incomplete_cancellation_is_not_success(
        #[case] wrong_hash: bool,
        #[case] after_batches: u64,
        #[case] message: &str,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("data.lance").to_str().unwrap().to_string();
        let expected = prepare(Fixture::FullZip, &uri, 17, 67, 5).await.unwrap();
        let dataset = Dataset::open(&uri).await.unwrap();
        let config = DrainConfig {
            mode: Mode::Prefetched,
            batch_rows: 2,
            workspace: 65536,
            attempt_workspace: None,
            slow: Duration::ZERO,
            requests: 3,
            pool_bytes: 65536,
        };
        let budget = AllocationBudget::new(config.pool_bytes);
        let hash = if wrong_hash {
            "wrong digest"
        } else {
            &expected.hash
        };
        let error = tokio::time::timeout(
            Duration::from_secs(5),
            run(&dataset, &config, &budget, 17, hash, after_batches, None),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(budget.reserved(), 0);
    }

    #[tokio::test]
    async fn command_reports_cancelled_prefix_separately_from_complete_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("data.lance").to_str().unwrap().to_string();
        let expected = prepare(Fixture::FullZip, &uri, 17, 67, 5).await.unwrap();
        let args = [
            "run-cancel",
            "prefetched",
            &uri,
            "17",
            &expected.hash,
            "2",
            "65536",
            "0",
            "3",
            "65536",
            "1",
        ]
        .map(String::from)
        .to_vec();
        // This is a synthetic command-test identity, not a measured artifact SHA.
        let source_head = "0123456789abcdef0123456789abcdef01234567";
        let result =
            tokio::time::timeout(Duration::from_secs(5), run_command(args, Some(source_head)))
                .await
                .unwrap()
                .unwrap();
        assert_eq!(result["source_head"], source_head);
        assert_eq!(result["requests"], 3);
        assert_eq!(result["final_reserved_bytes"], 0);
        let outputs = result["outputs"].as_array().unwrap();
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            assert_eq!(output["rows"], 17);
            assert_eq!(output["hash"], expected.hash);
            assert_eq!(output["payload_bytes"], expected.payload_bytes);
            assert!(output["completed_seconds"].is_number());
        }
        assert_eq!(result["cancelled"]["request_index"], 0);
        assert_eq!(result["cancelled"]["prefix"]["batches"], 1);
        assert!(result["cancelled"]["prefix"]["completed_seconds"].is_null());
        assert!(result["cancelled"]["observed_seconds"].is_number());
    }

    #[rstest]
    #[case::missing(None)]
    #[case::short(Some("01234567"))]
    #[case::non_hex(Some("0123456789abcdef0123456789abcdef0123456g"))]
    #[tokio::test]
    async fn command_rejects_invalid_source_before_open(#[case] source_head: Option<&str>) {
        let args = [
            "run-cancel",
            "prefetched",
            "unused",
            "17",
            "hash",
            "2",
            "65536",
            "0",
            "3",
            "65536",
            "1",
        ]
        .map(String::from)
        .to_vec();
        let error = run_command(args, source_head).await.unwrap_err();
        assert!(
            error.to_string().contains("requires T210_SOURCE_HEAD"),
            "{error}"
        );
    }
}
