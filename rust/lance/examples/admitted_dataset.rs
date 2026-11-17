// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Source-matched ordinary/admitted Dataset benchmark, for managed VMs only.
//!
//! `prepare URI ROWS FRAME_BYTES FRAGMENT_ROWS` creates one immutable fixture and
//! prints its complete ordered payload hash. URI may be an authorized S3 prefix.
//! `prepare-dictionary` accepts the same arguments and generates 31 repeated
//! frame values with the writer's default index and dictionary compression.
//! It verifies that every stored frames page is MiniBlock with an LZ4 dictionary.
//! A short final fragment or unsuitable page size can fail this check; that
//! fixture is not qualified and is not overwritten on a retry. Both workloads
//! use the same run/run-many commands; fixture validation is outside measured reads.
//! `run MODE URI EXPECTED_ROWS EXPECTED_HASH BATCH_ROWS WORKSPACE_BYTES SLOW_US`
//! drains it without collecting batches. MODE is `ordinary` or `admitted`.
//! `run-many MODE URI EXPECTED_ROWS EXPECTED_HASH BATCH_ROWS WORKSPACE_BYTES SLOW_US REQUESTS POOL_BYTES`
//! runs finite independent complete drains against one immutable snapshot. Admitted
//! requests share one payload pool, while each mode keeps the same per-request I/O
//! budget. The configured I/O budget therefore scales with REQUESTS; the payload
//! pool is not an RSS or total transport-memory bound. Each request owns one sink.
//! Per-request first-batch and completion times share the group start (including
//! dataset open and admission delay). Credit samples include other active requests
//! and workspace headroom, not just that batch and not an allocation peak trace.
//! A capacity refusal or wrong count/order/hash is a failed run, never partial
//! success. Run each mode in a fresh process with the same cgroup, cache state,
//! source, fixture, batch ceiling and I/O budget. Hashing and optional consumer
//! delay are included in elapsed time. This is a Lance-path experiment, not
//! PE/RPC/QN, external HTTP/IPC sink, or process-wide server acceptance.
//!
//! Build with `T210_SOURCE_HEAD=<verified full SHA>` and the pinned
//! `release-with-debug` profile on a managed VM. Correctness tests below run in
//! CI without timing or process-memory measurement. Never run this benchmark on
//! the developer laptop. No default scanner, dependency or format is changed.

use std::{
    collections::HashMap,
    error::Error,
    io::Write,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{
    Array, RecordBatch, RecordBatchIterator,
    builder::{LargeBinaryBuilder, ListBuilder},
    cast::AsArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::{
    Dataset,
    dataset::{AdmittedDatasetScanOutcome, WriteParams, fragment::FragReadConfig},
};
use lance_core::utils::allocation::AllocationBudget;
use lance_file::version::LanceFileVersion;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use serde::Serialize;

#[path = "admitted_dataset/dictionary.rs"]
mod dictionary;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone, Copy, Debug)]
enum Fixture {
    FullZip,
    Dictionary,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Ordinary,
    Admitted,
}

#[derive(Serialize, Debug)]
struct Output {
    rows: u64,
    payload_bytes: u64,
    hash: String,
    batches: u64,
    shared_reserved_bytes_on_batch_max: usize,
    first_batch_seconds: Option<f64>,
    completed_seconds: Option<f64>,
}

struct Sink {
    hash: blake3::Hasher,
    rows: u64,
    bytes: u64,
    batches: u64,
    shared_reserved_bytes_on_batch_max: usize,
    started: Option<Instant>,
    first_batch_seconds: Option<f64>,
}

impl Sink {
    fn new(started: Option<Instant>) -> Self {
        Self {
            hash: blake3::Hasher::new(),
            rows: 0,
            bytes: 0,
            batches: 0,
            shared_reserved_bytes_on_batch_max: 0,
            started,
            first_batch_seconds: None,
        }
    }

    fn consume(&mut self, batch: &RecordBatch) -> Result<()> {
        let keys = batch
            .column_by_name("key")
            .ok_or("missing key")?
            .as_binary_opt::<i64>()
            .ok_or("key must be LargeBinary")?;
        let lists = batch
            .column_by_name("frames")
            .ok_or("missing frames")?
            .as_list_opt::<i32>()
            .ok_or("frames must be List")?;
        let values = lists
            .values()
            .as_binary_opt::<i64>()
            .ok_or("frames child must be LargeBinary")?;
        if self.batches == 0 {
            self.first_batch_seconds = self.started.map(|t| t.elapsed().as_secs_f64());
        }
        for row in 0..batch.num_rows() {
            if keys.is_null(row) || keys.value(row) != self.rows.to_le_bytes() {
                return Err(format!("out-of-order or missing physical row {}", self.rows).into());
            }
            self.hash.update(keys.value(row));
            self.bytes = self
                .bytes
                .checked_add(8)
                .ok_or("payload byte count overflow")?;
            self.hash.update(&[u8::from(lists.is_valid(row))]);
            if lists.is_valid(row) {
                let start = lists.value_offsets()[row] as usize;
                let end = lists.value_offsets()[row + 1] as usize;
                self.hash.update(&((end - start) as u64).to_le_bytes());
                for index in start..end {
                    self.hash.update(&[u8::from(values.is_valid(index))]);
                    if values.is_valid(index) {
                        let value = values.value(index);
                        self.hash.update(&(value.len() as u64).to_le_bytes());
                        self.hash.update(value);
                        self.bytes = self
                            .bytes
                            .checked_add(value.len() as u64)
                            .ok_or("payload byte count overflow")?;
                    }
                }
            }
            self.rows = self.rows.checked_add(1).ok_or("row count overflow")?;
        }
        self.batches = self.batches.checked_add(1).ok_or("batch count overflow")?;
        Ok(())
    }

    fn finish(self) -> Output {
        Output {
            rows: self.rows,
            payload_bytes: self.bytes,
            hash: self.hash.finalize().to_hex().to_string(),
            batches: self.batches,
            shared_reserved_bytes_on_batch_max: self.shared_reserved_bytes_on_batch_max,
            first_batch_seconds: self.first_batch_seconds,
            completed_seconds: self.started.map(|t| t.elapsed().as_secs_f64()),
        }
    }
}

fn frame_field(fixture: Fixture) -> Arc<Field> {
    let field = Field::new("item", DataType::LargeBinary, true);
    Arc::new(match fixture {
        Fixture::FullZip => field,
        Fixture::Dictionary => field.with_metadata(HashMap::from([
            (
                "lance-encoding:structural-encoding".into(),
                "miniblock".into(),
            ),
            ("lance-encoding:dict-divisor".into(), "1".into()),
        ])),
    })
}

fn fixture_schema(fixture: Fixture) -> Arc<Schema> {
    let fullzip = |name, data_type, nullable| {
        Field::new(name, data_type, nullable).with_metadata(HashMap::from([(
            "lance-encoding:structural-encoding".into(),
            "fullzip".into(),
        )]))
    };
    Arc::new(Schema::new(vec![
        fullzip("key", DataType::LargeBinary, false),
        Field::new("frames", DataType::List(frame_field(fixture)), true).with_metadata(
            HashMap::from([(
                "lance-encoding:structural-encoding".into(),
                match fixture {
                    Fixture::FullZip => "fullzip",
                    Fixture::Dictionary => "miniblock",
                }
                .into(),
            )]),
        ),
    ]))
}

fn fixture_batch(
    fixture: Fixture,
    start: u64,
    count: usize,
    width: usize,
) -> std::result::Result<RecordBatch, arrow_schema::ArrowError> {
    let mut keys = LargeBinaryBuilder::new();
    let mut lists = ListBuilder::new(LargeBinaryBuilder::new()).with_field(frame_field(fixture));
    for row in start..start + count as u64 {
        keys.append_value(row.to_le_bytes());
        if row % 17 == 0 {
            lists.append(false);
        } else if row % 19 == 0 {
            lists.append(true);
        } else {
            for frame in 0u64..2 {
                if frame == 1 && row % 23 == 0 {
                    lists.values().append_null();
                } else {
                    let mut bytes = vec![0; width];
                    match fixture {
                        Fixture::FullZip => {
                            // Preserve the original incompressible image-like fixture.
                            let mut hash = blake3::Hasher::new();
                            hash.update(&row.to_le_bytes());
                            hash.update(&frame.to_le_bytes());
                            hash.finalize_xof().fill(&mut bytes);
                        }
                        Fixture::Dictionary => {
                            // Repeat 31 distinct frames. Each frame repeats a 32-byte
                            // pattern, so dictionary-value compression has real work.
                            let value = ((row % 31) * 2 + frame) % 31;
                            let pattern = blake3::hash(&value.to_le_bytes());
                            for (index, byte) in bytes.iter_mut().enumerate() {
                                *byte = pattern.as_bytes()[index % 32];
                            }
                        }
                    }
                    lists.values().append_value(bytes);
                }
            }
            lists.append(true);
        }
    }
    RecordBatch::try_new(
        fixture_schema(fixture),
        vec![Arc::new(keys.finish()), Arc::new(lists.finish())],
    )
}

async fn prepare(
    fixture: Fixture,
    uri: &str,
    rows: u64,
    width: usize,
    fragment_rows: usize,
) -> Result<Output> {
    if rows == 0 || width == 0 || fragment_rows == 0 {
        return Err("rows, frame bytes and fragment rows must be positive".into());
    }
    // Fixture generation itself stays bounded. This is outside measured reads.
    let batch_rows =
        (8 * 1024 * 1024 / width.checked_mul(2).ok_or("frame width overflow")?).clamp(1, 256);
    let batches = (0..rows).step_by(batch_rows).map(move |start| {
        fixture_batch(
            fixture,
            start,
            (rows - start).min(batch_rows as u64) as usize,
            width,
        )
    });
    let dataset = Dataset::write(
        RecordBatchIterator::new(batches, fixture_schema(fixture)),
        uri,
        Some(WriteParams {
            max_rows_per_file: fragment_rows,
            max_rows_per_group: batch_rows,
            data_storage_version: Some(LanceFileVersion::V2_3),
            ..Default::default()
        }),
    )
    .await?;
    if matches!(fixture, Fixture::Dictionary) {
        dictionary::verify_layout(&dataset).await?;
    }
    // Regenerate the expected logical stream independently of all file readers.
    // This bounded second pass is fixture preparation, never measured scan time.
    let mut expected = Sink::new(None);
    for start in (0..rows).step_by(batch_rows) {
        expected.consume(&fixture_batch(
            fixture,
            start,
            (rows - start).min(batch_rows as u64) as usize,
            width,
        )?)?;
    }
    Ok(expected.finish())
}

struct DrainConfig {
    mode: Mode,
    batch_rows: u32,
    workspace: usize,
    slow: Duration,
    requests: usize,
    pool_bytes: usize,
}

async fn drain(
    dataset: &Dataset,
    config: &DrainConfig,
    budget: &Arc<AllocationBudget>,
    started: Option<Instant>,
) -> Result<Output> {
    let DrainConfig {
        mode,
        batch_rows,
        workspace,
        slow,
        ..
    } = *config;
    if batch_rows == 0 || workspace == 0 {
        return Err("batch rows and workspace must be positive".into());
    }
    let mut sink = Sink::new(started);
    match mode {
        Mode::Ordinary => {
            let mut scanner = dataset.scan();
            scanner
                .batch_size(batch_rows as usize)
                .batch_readahead(1)
                .fragment_readahead(1)
                .io_buffer_size(workspace as u64)
                .scan_in_order(true);
            let mut stream = scanner.try_into_stream().await?;
            while let Some(batch) = stream.try_next().await? {
                sink.consume(&batch)?;
                if !slow.is_zero() {
                    tokio::time::sleep(slow).await;
                }
                if config.requests > 1 {
                    tokio::task::yield_now().await;
                }
                drop(batch);
            }
        }
        Mode::Admitted => {
            let scheduler = ScanScheduler::new(
                dataset.object_store(None).await?,
                SchedulerConfig::new(workspace as u64),
            );
            let read_config = FragReadConfig::default().with_scan_scheduler(scheduler);
            let mut scan = dataset.try_scan_admitted_fullzip(
                dataset.schema().clone(),
                read_config,
                batch_rows,
                workspace,
                budget.clone(),
            )?;
            loop {
                match scan.next_when_available().await? {
                    AdmittedDatasetScanOutcome::Batch { batch, .. } => {
                        sink.shared_reserved_bytes_on_batch_max = sink
                            .shared_reserved_bytes_on_batch_max
                            .max(budget.reserved());
                        sink.consume(&batch)?;
                        if !slow.is_zero() {
                            tokio::time::sleep(slow).await;
                        }
                        if config.requests > 1 {
                            tokio::task::yield_now().await;
                        }
                        drop(batch);
                    }
                    AdmittedDatasetScanOutcome::NeedsCapacity => {
                        return Err(format!(
                            "incomplete at {:?}: a minimal attempt did not fit {workspace} bytes",
                            scan.position()
                        )
                        .into());
                    }
                    AdmittedDatasetScanOutcome::Finished => break,
                }
            }
        }
    }
    Ok(sink.finish())
}

// Drop every request future on failure, then let its already-dispatched I/O/CPU
// release owners before returning. Only fixed-size summaries are collected.
async fn drain_all(
    dataset: &Dataset,
    config: &DrainConfig,
    budget: &Arc<AllocationBudget>,
    expected_rows: u64,
    expected_hash: &str,
    started: Option<Instant>,
) -> Result<Vec<Output>> {
    if config.requests == 0 || config.pool_bytes == 0 {
        return Err("requests and pool bytes must be positive".into());
    }
    let result: Result<Vec<Output>> =
        futures::future::try_join_all((0..config.requests).map(|_| async move {
            let output = drain(dataset, config, budget, started).await?;
            verify(&output, expected_rows, expected_hash)?;
            Ok(output)
        }))
        .await;
    // No result owns a batch and every request future has completed or dropped.
    // This is cleanup of dispatched work, not a wait inside a dependent stage.
    budget.wait_for_capacity(config.pool_bytes).await?;
    if budget.reserved() != 0 {
        return Err("completed request group retained allocation credit".into());
    }
    result
}

fn verify(output: &Output, rows: u64, hash: &str) -> Result<()> {
    if output.rows != rows || output.hash != hash {
        return Err(format!(
            "incomplete or altered output: expected {rows}/{hash}, got {}/{}",
            output.rows, output.hash
        )
        .into());
    }
    Ok(())
}

fn peak_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("prepare" | "prepare-dictionary") if args.len() == 5 => {
            let fixture = if args[0] == "prepare-dictionary" { Fixture::Dictionary } else { Fixture::FullZip };
            let output = prepare(fixture, &args[1], args[2].parse()?, args[3].parse()?, args[4].parse()?).await?;
            serde_json::json!({"preparation":args[0],"fixture":args[1],"rows":output.rows,"frame_bytes":args[3],"fragment_rows":args[4],"expected":output})
        }
        Some("run" | "run-many") if (args[0] == "run" && args.len() == 8) || (args[0] == "run-many" && args.len() == 10) => {
            let source_head = option_env!("T210_SOURCE_HEAD")
                .filter(|value| value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit()))
                .ok_or("benchmark build requires T210_SOURCE_HEAD with the verified full source SHA")?;
            let mode = match args[1].as_str() { "ordinary" => Mode::Ordinary, "admitted" => Mode::Admitted, _ => return Err("mode must be ordinary or admitted".into()) };
            let workspace = args[6].parse()?;
            let config = DrainConfig {
                mode, workspace, batch_rows: args[5].parse()?, slow: Duration::from_micros(args[7].parse()?),
                requests: if args[0] == "run-many" { args[8].parse()? } else { 1 },
                pool_bytes: if args[0] == "run-many" { args[9].parse()? } else { workspace },
            };
            let io_total = workspace.checked_mul(config.requests).ok_or("configured I/O budgets overflow")?;
            let budget = AllocationBudget::new(config.pool_bytes);
            let started = Instant::now();
            let dataset = Dataset::open(&args[2]).await?;
            let mut outputs = drain_all(&dataset, &config, &budget, args[3].parse()?, &args[4], Some(started)).await?;
            let mut result = serde_json::json!({"mode":args[1],"uri":args[2],"source_head":source_head,
                "batch_rows":args[5],"workspace_bytes":workspace,"slow_us":args[7],
                "requests":config.requests,"shared_payload_budget_bytes":config.pool_bytes,
                "io_budget_per_request_bytes":workspace,"io_budget_configured_sum_bytes":io_total,
                "elapsed_seconds":started.elapsed().as_secs_f64(),"peak_rss_kib":peak_rss_kib()});
            if args[0] == "run" {
                result["output"] = serde_json::to_value(outputs.pop().ok_or("missing complete drain result")?)?;
            } else {
                result["outputs"] = serde_json::to_value(outputs)?;
            }
            result
        }
        _ => return Err("usage: prepare|prepare-dictionary URI ROWS FRAME_BYTES FRAGMENT_ROWS | run MODE URI EXPECTED_ROWS EXPECTED_HASH BATCH_ROWS WORKSPACE_BYTES SLOW_US | run-many MODE URI EXPECTED_ROWS EXPECTED_HASH BATCH_ROWS WORKSPACE_BYTES SLOW_US REQUESTS POOL_BYTES".into()),
    };
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &result)?;
    writeln!(stdout)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::LargeBinaryArray;
    use rstest::rstest;

    #[rstest]
    #[case::short(55, 67, 7, 3)]
    #[case::wide(55, 4096, 13, 17)]
    #[tokio::test]
    async fn complete_outputs_match_prepared_hash_across_fragments(
        #[case] rows: u64,
        #[case] width: usize,
        #[case] fragment_rows: usize,
        #[case] batch_rows: u32,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("data.lance").to_str().unwrap().to_string();
        let expected = prepare(Fixture::FullZip, &uri, rows, width, fragment_rows)
            .await
            .unwrap();
        let dataset = Dataset::open(&uri).await.unwrap();
        for mode in [Mode::Ordinary, Mode::Admitted] {
            let config = DrainConfig {
                mode,
                batch_rows,
                workspace: 1024 * 1024,
                slow: Duration::ZERO,
                requests: 1,
                pool_bytes: 1024 * 1024,
            };
            let budget = AllocationBudget::new(config.pool_bytes);
            let outputs = drain_all(&dataset, &config, &budget, rows, &expected.hash, None)
                .await
                .unwrap();
            assert_eq!(outputs.len(), 1);
            let output = &outputs[0];
            assert_eq!(output.payload_bytes, expected.payload_bytes);
            assert!(output.batches > 1);
            assert_eq!(output.first_batch_seconds, None);
            assert_eq!(output.completed_seconds, None);
            assert!(output.shared_reserved_bytes_on_batch_max <= config.pool_bytes);
            assert_eq!(budget.reserved(), 0);
        }
    }

    #[tokio::test]
    async fn refusal_is_not_success_and_fixture_is_not_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("data.lance").to_str().unwrap().to_string();
        let expected = prepare(Fixture::FullZip, &uri, 3, 4096, 2).await.unwrap();
        let error = prepare(Fixture::FullZip, &uri, 3, 4096, 2)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("exist"));
        let dataset = Dataset::open(&uri).await.unwrap();
        let config = DrainConfig {
            mode: Mode::Admitted,
            batch_rows: 3,
            workspace: 1,
            slow: Duration::ZERO,
            requests: 2,
            pool_bytes: 1,
        };
        let budget = AllocationBudget::new(config.pool_bytes);
        let error = drain_all(&dataset, &config, &budget, 3, &expected.hash, None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("incomplete at"));
        assert_eq!(budget.reserved(), 0);
        let config = DrainConfig {
            mode: Mode::Ordinary,
            workspace: 65536,
            pool_bytes: 65536,
            ..config
        };
        let outputs = drain_all(
            &dataset,
            &config,
            &AllocationBudget::new(65536),
            3,
            &expected.hash,
            None,
        )
        .await
        .unwrap();
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn sink_rejects_reordered_rows_and_checks_all_payload_bytes() {
        let mut sink = Sink::new(None);
        let error = sink
            .consume(&fixture_batch(Fixture::FullZip, 1, 1, 32).unwrap())
            .unwrap_err();
        assert!(error.to_string().contains("out-of-order"));
        let mut original = Sink::new(None);
        original
            .consume(&fixture_batch(Fixture::FullZip, 0, 3, 32).unwrap())
            .unwrap();
        let original = original.finish();
        let batch = fixture_batch(Fixture::FullZip, 0, 3, 32).unwrap();
        let lists = batch["frames"].as_list::<i32>();
        let values = lists.values().as_binary::<i64>();
        let mut payloads = values
            .iter()
            .map(|value| value.map(<[u8]>::to_vec))
            .collect::<Vec<_>>();
        payloads[0].as_mut().unwrap()[16] ^= 1;
        let changed_lists = arrow_array::ListArray::new(
            Arc::new(Field::new("item", DataType::LargeBinary, true)),
            lists.offsets().clone(),
            Arc::new(LargeBinaryArray::from_iter(payloads)),
            lists.nulls().cloned(),
        );
        let changed_batch = RecordBatch::try_new(
            batch.schema(),
            vec![batch["key"].clone(), Arc::new(changed_lists)],
        )
        .unwrap();
        let mut changed = Sink::new(None);
        changed.consume(&changed_batch).unwrap();
        let changed = changed.finish();
        assert_eq!(changed.payload_bytes, original.payload_bytes);
        assert_ne!(changed.hash, original.hash);
        assert!(
            verify(&changed, original.rows, &original.hash)
                .unwrap_err()
                .to_string()
                .contains("altered output")
        );
        assert!(
            verify(&original, original.rows + 1, &original.hash)
                .unwrap_err()
                .to_string()
                .contains("incomplete")
        );
    }

    #[rstest]
    #[case::two_contending(2, 1)]
    #[case::three_contending(3, 1)]
    #[case::overlapping_workspaces(3, 2)]
    #[tokio::test]
    async fn concurrent_complete_drains_keep_independent_order_and_hash(
        #[case] requests: usize,
        #[case] workspace_slots: usize,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("data.lance").to_str().unwrap().to_string();
        let expected = prepare(Fixture::FullZip, &uri, 55, 4096, 7).await.unwrap();
        let dataset = Dataset::open(&uri).await.unwrap();
        for mode in [Mode::Ordinary, Mode::Admitted] {
            let config = DrainConfig {
                mode,
                batch_rows: 3,
                workspace: 65536,
                slow: Duration::ZERO,
                requests,
                pool_bytes: workspace_slots * 65536,
            };
            let budget = AllocationBudget::new(config.pool_bytes);
            let outputs = tokio::time::timeout(
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
            .expect("finite requests must complete even with one workspace and CPU worker")
            .unwrap();
            assert_eq!(outputs.len(), requests);
            for output in outputs {
                assert_eq!(output.rows, expected.rows);
                assert_eq!(output.hash, expected.hash);
                assert_eq!(output.payload_bytes, expected.payload_bytes);
                assert!(output.batches > 1);
                assert!(output.shared_reserved_bytes_on_batch_max <= config.pool_bytes);
                assert_eq!(output.first_batch_seconds, None);
                assert_eq!(output.completed_seconds, None);
            }
            assert_eq!(budget.reserved(), 0);
        }
    }

    #[tokio::test]
    async fn group_failure_and_cancel_release_credit_before_retry() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("data.lance").to_str().unwrap().to_string();
        let expected = prepare(Fixture::FullZip, &uri, 9, 67, 3).await.unwrap();
        let dataset = Dataset::open(&uri).await.unwrap();
        let config = DrainConfig {
            mode: Mode::Admitted,
            batch_rows: 2,
            workspace: 65536,
            slow: Duration::ZERO,
            requests: 3,
            pool_bytes: 65536,
        };
        let budget = AllocationBudget::new(config.pool_bytes);
        let error = tokio::time::timeout(
            Duration::from_secs(5),
            drain_all(
                &dataset,
                &config,
                &budget,
                expected.rows,
                "wrong digest",
                None,
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert!(error.to_string().contains("incomplete or altered output"));
        assert_eq!(budget.reserved(), 0);
        let held = budget.try_reserve(config.pool_bytes).unwrap();
        let mut pending = Box::pin(drain_all(
            &dataset,
            &config,
            &budget,
            expected.rows,
            &expected.hash,
            None,
        ));
        assert!(futures::poll!(pending.as_mut()).is_pending());
        drop(pending);
        assert_eq!(budget.reserved(), config.pool_bytes);
        drop(held);
        let outputs = tokio::time::timeout(
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
        assert_eq!(outputs.len(), config.requests);
        assert_eq!(budget.reserved(), 0);
    }

    #[rstest]
    #[case::no_requests(0, 65536, 65536)]
    #[case::no_pool(2, 65536, 0)]
    #[case::impossible_workspace(2, 65537, 65536)]
    #[tokio::test]
    async fn invalid_group_cannot_be_reported_as_complete(
        #[case] requests: usize,
        #[case] workspace: usize,
        #[case] pool_bytes: usize,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("data.lance").to_str().unwrap().to_string();
        let expected = prepare(Fixture::FullZip, &uri, 3, 67, 2).await.unwrap();
        let dataset = Dataset::open(&uri).await.unwrap();
        let config = DrainConfig {
            mode: Mode::Admitted,
            batch_rows: 2,
            workspace,
            slow: Duration::ZERO,
            requests,
            pool_bytes,
        };
        let budget = AllocationBudget::new(pool_bytes);
        let error = tokio::time::timeout(
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
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains(if requests == 0 || pool_bytes == 0 {
                    "must be positive"
                } else {
                    "exceeds allocation budget"
                })
        );
        assert_eq!(budget.reserved(), 0);
    }
}
