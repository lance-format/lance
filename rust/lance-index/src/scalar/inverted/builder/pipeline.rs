// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in build-time merging of owned, compressed runs.
//!
//! The experiment is opt-in so its scheduling and layout can be measured against
//! the unchanged default. The producer threshold is separate from the output
//! target; queued runs and the merge input have their own reserved capacity.
//!
//! `LANCE_FTS_PIPELINED_MERGE=1` enables this path for v3 builds. The default is
//! unchanged. `LANCE_FTS_PIPELINE_MERGE_WORKERS` defaults to one; it is not
//! inferred from CPU count. Optional `RUN_MIB`, `TARGET_MIB` and `QUEUE_RUNS`
//! settings under the same `LANCE_FTS_PIPELINE_` prefix control the producer
//! flush trigger, final partition target and bounded queue count. Merge workers
//! are a maximum concurrency: memory admission can reduce the active count,
//! rather than reducing partition size to accommodate more workers.
//! The working set includes a conservative allocation-growth allowance and
//! decoded positions scratch, so its reservation exceeds the output target.
//! Input Arrow batches, tokenization temporaries, allocator retention and object
//! store buffers also contribute to RSS; this is not a process memory limit.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use arrow_array::RecordBatch;
use async_channel::{Receiver, Sender};
use futures::TryStreamExt;
use lance_core::utils::tokio::{spawn, spawn_cpu};
use lance_core::{Error, Result};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use tokio::task::JoinSet;

mod merge;

use super::{
    IndexStore, IndexWorker, IndexWorkerConfig, InnerBuilder, InvertedIndexBuilder,
    InvertedListFormatVersion, PartitionWriteTarget, SendableRecordBatchStream, TailPartition,
    WorkerOutput, document_coordinate_rank, resolve_num_workers,
};

const QUEUE_UNIT_BYTES: u64 = 1024;

#[derive(Clone)]
struct Stop(watch::Sender<bool>);

impl Default for Stop {
    fn default() -> Self {
        Self(watch::channel(false).0)
    }
}

impl Stop {
    fn cancel(&self) {
        self.0.send_replace(true);
    }

    async fn cancelled(&self) {
        let mut receiver = self.0.subscribe();
        let _ = receiver.wait_for(|stopped| *stopped).await;
    }

    fn is_cancelled(&self) -> bool {
        *self.0.borrow()
    }
}

struct CancelOnDrop(Stop);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub(super) async fn build(
    builder: &InvertedIndexBuilder,
    mut stream: SendableRecordBatchStream,
    store: &dyn IndexStore,
    config: Config,
) -> Result<WorkerOutput> {
    if builder.format_version != InvertedListFormatVersion::V3 {
        return Err(Error::not_supported("pipelined FTS merge requires v3"));
    }
    let workers = resolve_num_workers(&builder.params);
    log::info!(
        "FTS_PIPELINE_CONFIG workers={} requested_merge_workers={} merge_workers={} queue_runs={} run_trigger_bytes={} max_run_bytes={} output_target_bytes={} merge_working_set_bytes={} queue_bytes={}",
        workers,
        config.requested_merge_workers,
        config.merge_workers,
        config.queue_runs,
        config.run_memory_limit_bytes,
        config.max_run_bytes,
        config.output_target_bytes,
        config.merge_memory_limit_bytes,
        config.queue_memory_bytes,
    );
    let tokenizer = builder.params.build()?;
    let worker_config = IndexWorkerConfig {
        with_position: builder.params.with_position,
        format_version: builder.format_version,
        fragment_mask: builder.fragment_mask,
        token_set_format: builder.token_set_format,
        worker_memory_limit_bytes: config.run_memory_limit_bytes,
        block_size: builder.params.block_size,
        coordinate_rank: document_coordinate_rank(&stream.schema()),
    };
    let ids = builder.partition_id_allocator();
    let tokenized = Arc::new(AtomicU64::new(0));
    let progress = builder.progress.clone();
    let target = builder.partition_write_target();
    let store = store.clone_arc();
    let (runs, run_receiver) = RunSender::new(config);
    let stop = runs.stop.clone();
    let cancel_on_drop = CancelOnDrop(stop.clone());
    let final_merge = merge::FinalMerge::new(
        builder,
        store.clone(),
        target,
        config.total_memory_bytes,
        stop.clone(),
    );

    // Cancellation of the public future signals this supervisor rather than
    // aborting it. It must retain and join workers that own CPU work or writes:
    // dropping a spawn_cpu future cannot stop an already running CPU closure.
    let supervisor = spawn(async move {
        let (sender, receiver) = async_channel::bounded::<RecordBatch>(workers);
        let mut tasks = JoinSet::new();
        for _ in 0..workers {
            let tokenizer = tokenizer.clone();
            let store = store.clone();
            let ids = ids.clone();
            let receiver = receiver.clone();
            let runs = runs.clone();
            let stop = stop.clone();
            let progress = progress.clone();
            let tokenized = tokenized.clone();
            tasks.spawn(async move {
                let mut worker = IndexWorker::new(tokenizer, store, ids, worker_config).await?;
                worker.run_sender = Some(runs);
                loop {
                    let batch = tokio::select! {
                        biased;
                        _ = stop.cancelled() => return Ok(WorkerOutput::default()),
                        batch = receiver.recv() => match batch {
                            Ok(batch) => batch,
                            Err(_) => break,
                        },
                    };
                    let rows = batch.num_rows();
                    worker.process_batch(batch).await?;
                    let completed =
                        tokenized.fetch_add(rows as u64, Ordering::Relaxed) + rows as u64;
                    tokio::select! {
                        biased;
                        _ = stop.cancelled() => return Ok(WorkerOutput::default()),
                        result = progress.stage_progress("tokenize_docs", completed) => result?,
                    }
                    tokio::task::yield_now().await;
                }
                worker.finish().await
            });
        }
        drop(receiver);
        drop(runs);
        for worker in 0..config.merge_workers {
            let receiver = run_receiver.clone();
            let store = store.clone();
            let stop = stop.clone();
            tasks.spawn(async move {
                consume_worker(receiver, store, target, config, worker, stop).await
            });
        }
        drop(run_receiver);
        let dispatch_stop = stop.clone();
        tasks.spawn(async move {
            loop {
                let batch = tokio::select! {
                    biased;
                    _ = dispatch_stop.cancelled() => break,
                    batch = stream.try_next() => match batch? {
                        Some(batch) => batch,
                        None => break,
                    },
                };
                tokio::select! {
                    biased;
                    _ = dispatch_stop.cancelled() => break,
                    result = sender.send(batch) => if result.is_err() { break; },
                }
            }
            Result::Ok(WorkerOutput::default())
        });

        let (output, tails) = finish_workers(tasks, &stop).await?;
        let output = finish_tails(output, tails, store, target, config, &stop).await?;
        final_merge.finish(output).await
    });
    let output = supervisor.await?;
    drop(cancel_on_drop);
    output
}

async fn finish_workers(
    mut tasks: JoinSet<Result<WorkerOutput>>,
    stop: &Stop,
) -> Result<(WorkerOutput, Vec<InnerBuilder>)> {
    let mut result = WorkerOutput::default();
    let mut tails = Vec::new();
    let mut first_error = None;
    let mut prerequisite_error = None;
    // Ordinary failures drain gracefully: aborting async workers would
    // detach any CPU closures and posting encoders they are awaiting.
    while let Some(joined) = tasks.join_next().await {
        match joined.map_err(Error::from).and_then(|output| output) {
            Ok(output) if first_error.is_none() => {
                result.partitions.extend(output.partitions);
                result.files.extend(output.files);
                if let Some(tail) = output.tail_partition {
                    tails.push(tail.builder);
                }
            }
            Ok(_) => {}
            Err(error @ Error::PrerequisiteFailed { .. }) => {
                // Closing a receiver can wake a blocked sender before the
                // failing consumer's task result reaches this JoinSet.
                prerequisite_error.get_or_insert(error);
                stop.cancel();
            }
            Err(error) => {
                first_error.get_or_insert(error);
                stop.cancel();
            }
        }
    }
    if let Some(error) = first_error.or(prerequisite_error) {
        return Err(error);
    }
    result.partitions.sort_unstable();
    Ok((result, tails))
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Config {
    pub requested_merge_workers: usize,
    pub merge_workers: usize,
    // Producer bursts can require more buffering than merge concurrency.
    pub queue_runs: usize,
    pub queue_memory_bytes: u64,
    pub run_memory_limit_bytes: u64,
    pub max_run_bytes: u64,
    pub output_target_bytes: u64,
    pub total_memory_bytes: u64,
    // Includes allocation growth and positions scratch, not just the output.
    pub merge_memory_limit_bytes: u64,
}

impl Config {
    pub(super) fn from_env(total_budget: u64, workers: usize) -> Result<Option<Self>> {
        match std::env::var("LANCE_FTS_PIPELINED_MERGE") {
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Ok(value) if value == "0" => return Ok(None),
            Ok(value) if value == "1" => {}
            value => {
                return Err(Error::invalid_input(format!(
                    "LANCE_FTS_PIPELINED_MERGE must be 0 or 1, got {value:?}"
                )));
            }
        }
        let target = optional_mib("LANCE_FTS_PIPELINE_TARGET_MIB")?;
        let run = optional_mib("LANCE_FTS_PIPELINE_RUN_MIB")?;
        let merge_workers = match std::env::var("LANCE_FTS_PIPELINE_MERGE_WORKERS") {
            Err(std::env::VarError::NotPresent) => 1,
            Ok(value) => value.parse::<usize>().map_err(|error| {
                Error::invalid_input(format!("invalid merge worker count {value:?}: {error}"))
            })?,
            Err(error) => {
                return Err(Error::invalid_input(format!(
                    "invalid merge worker count: {error}"
                )));
            }
        };
        let queue_runs = match std::env::var("LANCE_FTS_PIPELINE_QUEUE_RUNS") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) => Some(value.parse::<usize>().map_err(|error| {
                Error::invalid_input(format!(
                    "LANCE_FTS_PIPELINE_QUEUE_RUNS must be a run count, got {value:?}: {error}"
                ))
            })?),
            Err(error) => {
                return Err(Error::invalid_input(format!(
                    "invalid LANCE_FTS_PIPELINE_QUEUE_RUNS: {error}"
                )));
            }
        };
        Self::new(
            total_budget,
            workers,
            merge_workers,
            target,
            run,
            queue_runs,
        )
        .map(Some)
    }

    pub(super) fn new(
        total_budget: u64,
        workers: usize,
        merge_workers: usize,
        target: Option<u64>,
        run: Option<u64>,
        queue_runs: Option<usize>,
    ) -> Result<Self> {
        if workers == 0 || merge_workers == 0 || queue_runs == Some(0) {
            return Err(Error::invalid_input(format!(
                "pipeline worker and queue counts must be positive: workers={workers}, merge_workers={merge_workers}, queue_runs={queue_runs:?}"
            )));
        }
        // Match the ordinary per-worker trigger when feasible. Even a single
        // merger needs room for its working set plus producers and the queue.
        // This target never depends on requested merger concurrency.
        let output_target_bytes =
            target.unwrap_or((total_budget / workers as u64).min(total_budget / 8));
        // Before the output trigger, retained inputs cost at most target +
        // target/4. The 4x growth estimate consumes 5x target; reserve another
        // target for decoded-position scratch. Larger scratch uses Full safely.
        let merge_memory_limit_bytes = output_target_bytes
            .checked_mul(6)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| Error::invalid_input("invalid FTS pipeline output target"))?;
        let minimum_run_bytes = match run {
            Some(0) => {
                return Err(Error::invalid_input(
                    "pipeline run trigger must be positive",
                ));
            }
            Some(bytes) => bytes
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(QUEUE_UNIT_BYTES - 1))
                .map(|bytes| bytes / QUEUE_UNIT_BYTES * QUEUE_UNIT_BYTES)
                .ok_or_else(|| Error::invalid_input("pipeline run reservation overflow"))?,
            None => QUEUE_UNIT_BYTES,
        };
        let fixed_slots = (workers as u64)
            .checked_add(queue_runs.unwrap_or(0) as u64)
            .ok_or_else(|| Error::invalid_input("pipeline worker count overflow"))?;
        // With no explicit queue count, each admitted merger gets one queue
        // slot as well as its independent current-input reservation.
        let per_merger = minimum_run_bytes
            .checked_mul(if queue_runs.is_some() { 1 } else { 2 })
            .and_then(|bytes| bytes.checked_add(merge_memory_limit_bytes))
            .ok_or_else(|| Error::invalid_input("pipeline merge reservation overflow"))?;
        let available = fixed_slots
            .checked_mul(minimum_run_bytes)
            .and_then(|bytes| total_budget.checked_sub(bytes))
            .unwrap_or(0);
        let effective_workers = (available / per_merger).min(merge_workers as u64) as usize;
        if effective_workers == 0 || minimum_run_bytes > output_target_bytes / 4 {
            return Err(Error::invalid_input(format!(
                "FTS pipeline output target {output_target_bytes} and run reservation {minimum_run_bytes} exceed total budget {total_budget} with workers={workers}, requested_merge_workers={merge_workers}, queue_runs={queue_runs:?}"
            )));
        }
        let queue_runs = queue_runs.unwrap_or(effective_workers);
        // Only admitted consumers are spawned. A waiting extra consumer must
        // not depend on a slot retained by a completed consumer's tail.
        let remaining = total_budget - merge_memory_limit_bytes * effective_workers as u64;
        let slots = workers as u64 + effective_workers as u64 + queue_runs as u64;
        let max_run_bytes =
            (remaining / slots).min(output_target_bytes / 4) / QUEUE_UNIT_BYTES * QUEUE_UNIT_BYTES;
        let queue_memory_bytes = max_run_bytes
            .checked_mul(queue_runs as u64)
            .ok_or_else(|| Error::invalid_input("merge queue reservation overflow"))?;
        let run_memory_limit_bytes = run.unwrap_or(max_run_bytes / 2);
        if max_run_bytes < minimum_run_bytes
            || queue_memory_bytes / QUEUE_UNIT_BYTES > u32::MAX as u64
            || queue_memory_bytes / QUEUE_UNIT_BYTES > Semaphore::MAX_PERMITS as u64
        {
            return Err(Error::invalid_input(format!(
                "invalid FTS pipeline budgets: total={total_budget}, workers={workers}, merge_workers={effective_workers}, queue_runs={queue_runs}, output_target={output_target_bytes}, max_run={max_run_bytes}, run_trigger={run_memory_limit_bytes}"
            )));
        }
        Ok(Self {
            requested_merge_workers: merge_workers,
            merge_workers: effective_workers,
            queue_runs,
            queue_memory_bytes,
            run_memory_limit_bytes,
            max_run_bytes,
            output_target_bytes,
            total_memory_bytes: total_budget,
            merge_memory_limit_bytes,
        })
    }
}

fn optional_mib(name: &str) -> Result<Option<u64>> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok(value) => value
            .parse::<u64>()
            .ok()
            .and_then(|mib| mib.checked_mul(1 << 20))
            .map(Some)
            .ok_or_else(|| {
                Error::invalid_input(format!("{name} must be a MiB count, got {value:?}"))
            }),
        Err(error) => Err(Error::invalid_input(format!("invalid {name}: {error}"))),
    }
}

pub(super) struct Run {
    builder: InnerBuilder,
    permit: OwnedSemaphorePermit,
}

#[derive(Clone)]
pub(super) struct RunSender {
    sender: Sender<Run>,
    bytes: Arc<Semaphore>,
    max_run_bytes: u64,
    stop: Stop,
}

impl RunSender {
    pub(super) fn new(config: Config) -> (Self, Receiver<Run>) {
        let (sender, receiver) = async_channel::bounded(config.queue_runs);
        let bytes = Arc::new(Semaphore::new(
            (config.queue_memory_bytes / QUEUE_UNIT_BYTES) as usize,
        ));
        (
            Self {
                sender,
                bytes,
                max_run_bytes: config.max_run_bytes,
                stop: Stop::default(),
            },
            receiver,
        )
    }

    pub(super) async fn send(&self, builder: InnerBuilder) -> Result<()> {
        let size = builder.memory_size();
        if size > self.max_run_bytes {
            return Err(Error::invalid_input(format!(
                "FTS pipeline run {size} bytes exceeds its {}-byte reservation",
                self.max_run_bytes
            )));
        }
        let units = u32::try_from(size.div_ceil(QUEUE_UNIT_BYTES)).map_err(|error| {
            Error::invalid_input(format!("FTS pipeline run size {size}: {error}"))
        })?;
        let permit = tokio::select! {
            biased;
            _ = self.stop.cancelled() => {
                return Err(Error::prerequisite_failed("FTS pipeline cancelled"));
            }
            _ = self.sender.closed() => {
                return Err(Error::prerequisite_failed("FTS pipeline merge consumer closed"));
            }
            permit = self.bytes.clone().acquire_many_owned(units) => {
                permit.map_err(|error| Error::prerequisite_failed(format!("FTS pipeline queue closed: {error}")))?
            }
        };
        tokio::select! {
            biased;
            _ = self.stop.cancelled() => Err(Error::prerequisite_failed("FTS pipeline cancelled")),
            result = self.sender.send(Run { builder, permit }) => result
                .map_err(|_| Error::prerequisite_failed("FTS pipeline merge consumer closed")),
        }
    }
}

enum MergeOutcome {
    Merged(InnerBuilder),
    Full {
        previous: Box<InnerBuilder>,
        incoming: InnerBuilder,
    },
}

fn merge_runs(
    mut previous: InnerBuilder,
    incoming: InnerBuilder,
    budget: u64,
) -> Result<MergeOutcome> {
    let combined = previous
        .memory_size()
        .checked_add(incoming.memory_size())
        .ok_or_else(|| Error::invalid_input("FTS pipeline merge memory size overflow"))?;
    let documents = previous
        .docs
        .len()
        .checked_add(incoming.docs.len())
        .ok_or_else(|| Error::invalid_input("FTS pipeline document count overflow"))?;
    let scratch = incoming
        .posting_lists
        .iter()
        .try_fold(0, |maximum, posting| {
            posting
                .merge_scratch_bytes()
                .map(|bytes| maximum.max(bytes))
        })?;
    // Both compressed inputs, allocation growth, and positions scratch must fit.
    // Frequencies determine scratch size; compressed position bytes do not.
    let working_set = combined
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(scratch))
        .ok_or_else(|| Error::invalid_input("FTS merge working set overflow"))?;
    if working_set >= budget || documents > u32::MAX as usize {
        return Ok(MergeOutcome::Full {
            previous: Box::new(previous),
            incoming,
        });
    }
    previous.merge_from(incoming)?;
    Ok(MergeOutcome::Merged(previous))
}

async fn consume_worker(
    receiver: Receiver<Run>,
    store: Arc<dyn IndexStore>,
    target: PartitionWriteTarget,
    config: Config,
    worker: usize,
    stop: Stop,
) -> Result<WorkerOutput> {
    let started = Instant::now();
    let mut accumulated: Option<InnerBuilder> = None;
    let mut partitions = Vec::new();
    let mut files = Vec::new();
    let mut runs = 0usize;
    let mut merge_seconds = 0.0;
    let mut write_seconds = 0.0;
    let mut max_run_bytes = 0;
    let mut max_output_bytes = 0;
    let mut undersized_partitions = 0;
    loop {
        let Run { builder, permit } = tokio::select! {
            biased;
            _ = stop.cancelled() => return Ok(WorkerOutput::default()),
            run = receiver.recv() => match run {
                Ok(run) => run,
                Err(_) => break,
            },
        };
        // The consumer has a separate input reservation. Release queue credit
        // before CPU work or I/O so producers can continue filling the pipeline.
        drop(permit);
        runs += 1;
        max_run_bytes = max_run_bytes.max(builder.memory_size());
        accumulated = Some(match accumulated.take() {
            Some(previous) => {
                let merging = Instant::now();
                let outcome = spawn_cpu(move || {
                    merge_runs(previous, builder, config.merge_memory_limit_bytes)
                })
                .await?;
                merge_seconds += merging.elapsed().as_secs_f64();
                if stop.is_cancelled() {
                    return Ok(WorkerOutput::default());
                }
                match outcome {
                    MergeOutcome::Merged(merged) => merged,
                    MergeOutcome::Full { previous, incoming } => {
                        undersized_partitions +=
                            usize::from(previous.memory_size() < config.output_target_bytes);
                        let writing = Instant::now();
                        let (id, written) =
                            merge::write_run(*previous, store.as_ref(), target, &stop).await?;
                        files.extend(written);
                        partitions.push(id);
                        write_seconds += writing.elapsed().as_secs_f64();
                        incoming
                    }
                }
            }
            None => builder,
        });
        if let Some(builder) = &accumulated {
            max_output_bytes = max_output_bytes.max(builder.memory_size());
        }
        if let Some(builder) =
            accumulated.take_if(|builder| builder.memory_size() >= config.output_target_bytes)
        {
            let writing = Instant::now();
            let (id, written) = merge::write_run(builder, store.as_ref(), target, &stop).await?;
            files.extend(written);
            partitions.push(id);
            write_seconds += writing.elapsed().as_secs_f64();
        }
    }
    if stop.is_cancelled() {
        return Ok(WorkerOutput::default());
    }
    log::info!(
        "FTS_PIPELINE_WORKER_COMPLETE worker={} runs={} partitions={} merge_task_seconds={:.3} write_seconds={:.3} \
         undersized_partitions={} max_run_bytes={} max_output_bytes={} tail_bytes={} elapsed_seconds={:.3}",
        worker,
        runs,
        partitions.len(),
        merge_seconds,
        write_seconds,
        undersized_partitions,
        max_run_bytes,
        max_output_bytes,
        accumulated.as_ref().map_or(0, InnerBuilder::memory_size),
        started.elapsed().as_secs_f64()
    );
    Ok(WorkerOutput {
        partitions,
        files,
        tail_partition: accumulated.map(|builder| TailPartition { builder }),
    })
}

async fn finish_tails(
    mut output: WorkerOutput,
    mut tails: Vec<InnerBuilder>,
    store: Arc<dyn IndexStore>,
    target: PartitionWriteTarget,
    config: Config,
    stop: &Stop,
) -> Result<WorkerOutput> {
    // Producers and consumers are joined. Their static reservations remain
    // available to this sequential drain, except for the tails still retained.
    // Merge smallest tails first so a large tail cannot strand smaller ones.
    tails.sort_unstable_by_key(|builder| std::cmp::Reverse(builder.memory_size()));
    let mut idle_bytes = tails.iter().try_fold(0u64, |bytes, tail| {
        bytes
            .checked_add(tail.memory_size())
            .ok_or_else(|| Error::invalid_input("FTS pipeline tail memory overflow"))
    })?;
    let mut accumulated: Option<InnerBuilder> = None;
    let tail_count = tails.len();
    let previous_partitions = output.partitions.len();
    while let Some(incoming) = tails.pop() {
        if stop.is_cancelled() {
            return Ok(WorkerOutput::default());
        }
        idle_bytes -= incoming.memory_size();
        accumulated = Some(match accumulated.take() {
            Some(previous) => {
                let budget = config
                    .total_memory_bytes
                    .checked_sub(idle_bytes)
                    .ok_or_else(|| {
                        Error::invalid_input("FTS pipeline tails exceed total budget")
                    })?;
                let outcome = spawn_cpu(move || merge_runs(previous, incoming, budget)).await?;
                if stop.is_cancelled() {
                    return Ok(WorkerOutput::default());
                }
                match outcome {
                    MergeOutcome::Merged(builder) => builder,
                    MergeOutcome::Full { previous, incoming } => {
                        let (id, files) =
                            merge::write_run(*previous, store.as_ref(), target, stop).await?;
                        output.files.extend(files);
                        output.partitions.push(id);
                        incoming
                    }
                }
            }
            None => incoming,
        });
        if let Some(builder) =
            accumulated.take_if(|builder| builder.memory_size() >= config.output_target_bytes)
        {
            let (id, files) = merge::write_run(builder, store.as_ref(), target, stop).await?;
            output.files.extend(files);
            output.partitions.push(id);
        }
    }
    if !stop.is_cancelled()
        && let Some(builder) = accumulated
    {
        let (id, files) = merge::write_run(builder, store.as_ref(), target, stop).await?;
        output.files.extend(files);
        output.partitions.push(id);
    }
    log::info!(
        "FTS_PIPELINE_TAILS_COMPLETE tails={} partitions={}",
        tail_count,
        output.partitions.len() - previous_partitions
    );
    output.partitions.sort_unstable();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::inverted::index::TokenSetFormat;
    use crate::scalar::lance_format::LanceIndexStore;
    use arrow_array::cast::AsArray;
    use lance_core::cache::LanceCache;
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use std::sync::atomic::AtomicBool;
    use tokio::sync::oneshot;

    fn small_run(id: u64) -> InnerBuilder {
        let mut builder = InnerBuilder::new(id, false, TokenSetFormat::default());
        builder.tokens.get_or_add(&format!("token{id}"));
        let mut posting = super::super::PostingListBuilder::new(false);
        for row in 0..128 {
            let doc = builder.docs.append(id * 128 + row, 1);
            posting.add(doc, super::super::PositionRecorder::Count(1));
        }
        builder.posting_lists.push(posting);
        builder
    }

    #[tokio::test]
    async fn test_pipeline_output_target_is_not_the_working_set_limit() -> Result<()> {
        let target_bytes = 64 << 10;
        let config = Config::new(2 << 20, 1, 1, Some(target_bytes), None, None)?;
        let builders = (0..8).map(small_run).collect::<Vec<_>>();
        let retained = builders.iter().map(InnerBuilder::memory_size).sum::<u64>();
        assert!(retained < target_bytes && 4 * retained > target_bytes);
        let (sender, receiver) = RunSender::new(config);
        let producer = tokio::spawn(async move {
            for builder in builders {
                sender.send(builder).await?;
            }
            Result::Ok(())
        });
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::memory()),
            Path::from("index"),
            Arc::new(LanceCache::no_cache()),
        ));
        let output = consume_worker(
            receiver,
            store,
            PartitionWriteTarget::Final,
            config,
            0,
            Stop::default(),
        )
        .await?;
        producer.await??;
        assert_eq!(
            output.partitions.len() + usize::from(output.tail_partition.is_some()),
            1,
            "runs that fit one final partition must not be cut at target / 4"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_pipeline_drains_all_merger_tails_together() -> Result<()> {
        let config = Config::new(2 << 20, 1, 3, Some(64 << 10), None, None)?;
        assert_eq!(config.merge_workers, 3);
        let mut tasks = JoinSet::new();
        for id in 0..3 {
            tasks.spawn(async move {
                Ok(WorkerOutput {
                    tail_partition: Some(TailPartition {
                        builder: small_run(id),
                    }),
                    ..Default::default()
                })
            });
        }
        let stop = Stop::default();
        let (output, tails) = finish_workers(tasks, &stop).await?;
        assert_eq!(tails.len(), 3);
        assert!(output.files.is_empty() && output.partitions.is_empty());
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::memory()),
            Path::from("index"),
            Arc::new(LanceCache::no_cache()),
        ));
        let output = finish_tails(
            output,
            tails,
            store.clone(),
            PartitionWriteTarget::Final,
            config,
            &stop,
        )
        .await?;
        assert_eq!(output.partitions.len(), 1);
        let docs = store
            .open_index_file(&super::super::doc_file_path(output.partitions[0]))
            .await?;
        assert_eq!(docs.num_rows(), 384);
        let batch = docs.read_range(0..384, None).await?;
        let mut rows = batch[lance_core::ROW_ID]
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .to_vec();
        rows.sort_unstable();
        assert_eq!(rows, (0..384).collect::<Vec<u64>>());
        Ok(())
    }

    #[tokio::test]
    async fn test_pipeline_closed_channel_does_not_hide_consumer_error() -> Result<()> {
        let mut tasks = JoinSet::new();
        tasks.spawn(async {
            Result::<WorkerOutput>::Err(Error::prerequisite_failed("consumer closed"))
        });
        tasks.spawn(async {
            tokio::task::yield_now().await;
            Result::<WorkerOutput>::Err(Error::io("injected original write error"))
        });
        let error = finish_workers(tasks, &Stop::default())
            .await
            .err()
            .expect("the write failure must be returned");
        assert!(matches!(error, Error::IO { .. }), "{error}");
        assert!(error.to_string().contains("injected original write error"));
        Ok(())
    }

    #[tokio::test]
    async fn test_pipeline_failure_waits_for_inflight_cpu() -> Result<()> {
        let mut tasks = JoinSet::new();
        let (started, running) = oneshot::channel();
        let completed = Arc::new(AtomicBool::new(false));
        let cpu_completed = completed.clone();
        let owner = Arc::new(());
        let weak_owner = Arc::downgrade(&owner);
        tasks.spawn(async move {
            spawn_cpu(move || {
                // Notify the failing async sibling, then do fixed CPU work.
                // The CPU pool never waits on a channel or test barrier.
                let _ = started.send(());
                let mut state = 0u64;
                for value in 0..2_000_000 {
                    state = std::hint::black_box(state.rotate_left(7) ^ value);
                }
                std::hint::black_box(state);
                drop(owner);
                cpu_completed.store(true, Ordering::SeqCst);
                Result::Ok(())
            })
            .await?;
            Result::Ok(WorkerOutput::default())
        });
        tasks.spawn(async move {
            running.await.expect("CPU work must start");
            Err(Error::io("injected failure during CPU merge"))
        });
        let stop = Stop::default();
        let error = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            finish_workers(tasks, &stop),
        )
        .await
        .expect("supervisor must drain fixed CPU work")
        .err()
        .expect("the original failure must be returned");
        assert!(matches!(error, Error::IO { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("injected failure during CPU merge")
        );
        assert!(completed.load(Ordering::SeqCst));
        assert!(weak_owner.upgrade().is_none());
        Ok(())
    }

    #[test]
    fn test_pipeline_reserves_decoded_positions_before_merging() -> Result<()> {
        let mut previous = InnerBuilder::new_with_format_version_and_block_size(
            0,
            true,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V3,
            256,
        );
        previous.docs.append(0, 0);
        let mut incoming = InnerBuilder::new_with_format_version_and_block_size(
            1,
            true,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V3,
            256,
        );
        incoming.tokens.get_or_add("repeated");
        let doc = incoming.docs.append(1, 65_536);
        let mut posting =
            super::super::PostingListBuilder::new_with_posting_tail_codec_and_block_size(
                true,
                incoming.posting_tail_codec,
                256,
            );
        posting.add(
            doc,
            super::super::PositionRecorder::Position((0..65_536).collect()),
        );
        incoming.posting_lists.push(posting);
        let budget = 512 << 10;
        assert!(4 * (previous.memory_size() + incoming.memory_size()) < budget);
        // The compressed runs fit, but decoding all positions in this posting
        // would exceed the merger's working set. Keep both valid partitions.
        assert!(matches!(
            merge_runs(previous, incoming, budget)?,
            MergeOutcome::Full { .. }
        ));
        Ok(())
    }

    #[tokio::test]
    async fn test_pipeline_merge_failure_releases_queued_runs() -> Result<()> {
        let config = Config::new(256 << 10, 1, 1, None, None, Some(2))?;
        let (sender, receiver) = RunSender::new(config);
        let credit = sender.bytes.clone();
        let stop = sender.stop.clone();
        for (id, positions) in [(0, false), (1, true)] {
            let mut builder = InnerBuilder::new(id, positions, TokenSetFormat::default());
            builder.docs.append(id, 0);
            sender.send(builder).await?;
        }
        drop(sender);
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::memory()),
            Path::from("index"),
            Arc::new(LanceCache::no_cache()),
        ));
        let error = consume_worker(
            receiver,
            store.clone(),
            PartitionWriteTarget::Final,
            config,
            0,
            stop,
        )
        .await
        .err()
        .expect("merging incompatible runs must fail");
        assert!(matches!(error, Error::Index { .. }), "{error}");
        assert!(error.to_string().contains("mismatched positions"));
        assert_eq!(
            credit.available_permits(),
            (config.queue_memory_bytes / QUEUE_UNIT_BYTES) as usize
        );
        assert!(store.list_files_with_sizes().await?.is_empty());
        Ok(())
    }
}
