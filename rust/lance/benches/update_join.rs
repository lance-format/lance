// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Process-isolated benchmarks for fragment update joins.
//!
//! The default smoke suite is intentionally small. The threshold and core suites use the
//! production-scale row counts documented in `update_join_benchmark_plan.md`.
//!
//! ```text
//! cargo bench --profile release-with-debug -p lance \
//!   --features update-join-bench --bench update_join -- --suite smoke
//! cargo bench --profile release-with-debug -p lance \
//!   --features update-join-bench --bench update_join -- --suite thresholds --repetitions 5
//! ```

#![allow(clippy::print_stderr, clippy::print_stdout)]

use std::collections::HashMap;
use std::error::Error as StdError;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchReader,
    StringArray, UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema as ArrowSchema, SchemaRef};
use clap::{Parser, ValueEnum};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::fragment::FileFragment;
use lance::dataset::update_join_bench::{UpdateJoinBenchmarkOptions, update_columns};
use lance::dataset::{
    UpdateJoinOptions, UpdateJoinStrategy, WriteParams, fragment::FragmentUpdateColumnsResult,
};
use lance_core::utils::tempfile::TempStrDir;
use lance_datafusion::exec::{ExecutionSummaryCounts, LanceExecutionOptions};
use serde::Serialize;

const KEY_COLUMN: &str = "key";
const VALUE_COLUMN: &str = "value";
const DEFAULT_BATCH_SIZE: usize = 4_096;
const WIDE_DIMENSION: i32 = 1_024;
// Selector estimate: 4096-byte payload + 8-byte key + 2 * key buffers + 64 bytes/row.
const WIDE_RHS_ESTIMATED_BYTES_PER_ROW: u64 = 4_184;
const SHUFFLE_SEED: u64 = 0xD1B5_4A32_D192_ED03;
const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

type BenchResult<T> = Result<T, Box<dyn StdError + Send + Sync>>;

#[derive(Debug, Parser)]
#[command(about = "Compare hash and sort-merge fragment update joins")]
struct Args {
    /// Cargo passes this flag to benchmark executables.
    #[arg(long = "bench", hide = true)]
    _bench: bool,

    /// Predefined collection of scenarios to execute.
    #[arg(long, value_enum, default_value_t = Suite::Smoke)]
    suite: Suite,

    /// Run one scenario instead of a predefined suite.
    #[arg(long, value_enum)]
    scenario: Option<ScenarioName>,

    /// Strategy to run. `all` runs both algorithms and Auto on threshold scenarios.
    #[arg(long, value_enum, default_value_t = StrategyArg::All)]
    strategy: StrategyArg,

    /// Number of reported process-isolated runs per scenario and strategy.
    #[arg(long, default_value_t = 3)]
    repetitions: usize,

    /// Number of unreported process-isolated warm-up runs.
    #[arg(long, default_value_t = 1)]
    warmups: usize,

    /// DataFusion memory-pool size used by the external path.
    #[arg(long, default_value_t = 256)]
    memory_pool_mib: u64,

    /// Row count per DataFusion execution batch in the external path.
    #[arg(long, default_value_t = 256)]
    execution_batch_size: usize,

    /// Memory reserved for each external-sort merge.
    #[arg(long, default_value_t = 10)]
    sort_spill_reservation_mib: u64,

    /// Maximum temporary spill-directory size.
    #[arg(long, default_value_t = 100)]
    max_temp_gib: u64,

    /// Override the Auto row threshold. Must be supplied with `max_hash_mib`.
    #[arg(long)]
    max_hash_rows: Option<usize>,

    /// Override the Auto estimated-byte threshold. Must be supplied with `max_hash_rows`.
    #[arg(long)]
    max_hash_mib: Option<u64>,

    /// Execute one workload in this process. Used by the parent benchmark runner.
    #[arg(long, hide = true)]
    worker: bool,

    /// Repetition identifier included in the worker's result.
    #[arg(long, hide = true, default_value_t = 0)]
    repetition: usize,

    /// Mark this worker invocation as an unreported warm-up.
    #[arg(long, hide = true)]
    warmup: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Suite {
    Smoke,
    Thresholds,
    Core,
    Scaling,
    WideMemory,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum StrategyArg {
    All,
    Auto,
    Hash,
    SortMerge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ScenarioName {
    SmokeNarrow,
    SmallNarrow,
    RowBelow,
    RowAt,
    RowAbove,
    LargeDense,
    LargeSparse,
    WideBelow,
    WideAt,
    WideAbove,
    StringKeys,
    Presorted,
    Scale250k,
    Scale1m,
    Scale2m,
    Scale5m,
    Wide520,
    Wide1024,
    Wide2048,
    Wide4096,
    Wide8192,
    Wide16384,
}

#[derive(Clone, Copy, Debug)]
enum KeyKind {
    UInt64,
    String32,
}

#[derive(Clone, Copy, Debug)]
enum PayloadKind {
    Narrow,
    Wide,
}

#[derive(Clone, Copy, Debug)]
enum InputOrder {
    Sorted,
    Shuffled,
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: ScenarioName,
    left_rows: u64,
    rhs_rows: u64,
    match_percent: u64,
    key_kind: KeyKind,
    payload_kind: PayloadKind,
    left_order: InputOrder,
    rhs_order: InputOrder,
}

impl Scenario {
    fn matched_rows(self) -> u64 {
        self.left_rows
            .saturating_mul(self.match_percent)
            .checked_div(100)
            .unwrap_or_default()
            .min(self.rhs_rows)
    }

    fn is_threshold(self) -> bool {
        matches!(
            self.name,
            ScenarioName::RowBelow
                | ScenarioName::RowAt
                | ScenarioName::RowAbove
                | ScenarioName::WideBelow
                | ScenarioName::WideAt
                | ScenarioName::WideAbove
        )
    }

    fn target_rhs_estimated_mib(self) -> Option<u64> {
        match self.name {
            ScenarioName::Wide520 => Some(520),
            ScenarioName::Wide1024 => Some(1_024),
            ScenarioName::Wide2048 => Some(2_048),
            ScenarioName::Wide4096 => Some(4_096),
            ScenarioName::Wide8192 => Some(8_192),
            ScenarioName::Wide16384 => Some(16_384),
            _ => None,
        }
    }
}

impl ScenarioName {
    fn config(self) -> Scenario {
        let narrow = |left_rows, rhs_rows, match_percent| Scenario {
            name: self,
            left_rows,
            rhs_rows,
            match_percent,
            key_kind: KeyKind::UInt64,
            payload_kind: PayloadKind::Narrow,
            left_order: InputOrder::Sorted,
            rhs_order: InputOrder::Shuffled,
        };
        let wide_memory = |estimated_mib: u64| {
            let rows = estimated_mib
                .checked_mul(MIB)
                .expect("wide benchmark estimate fits u64")
                .div_ceil(WIDE_RHS_ESTIMATED_BYTES_PER_ROW);
            Scenario {
                payload_kind: PayloadKind::Wide,
                ..narrow(rows, rows, 100)
            }
        };
        match self {
            Self::SmokeNarrow => narrow(50_000, 20_000, 40),
            Self::SmallNarrow => narrow(1_000_000, 100_000, 10),
            Self::RowBelow => narrow(2_000_000, 900_000, 45),
            Self::RowAt => narrow(2_000_000, 1_000_000, 50),
            Self::RowAbove => narrow(2_000_000, 1_100_000, 50),
            Self::LargeDense => narrow(5_000_000, 5_000_000, 100),
            Self::LargeSparse => narrow(5_000_000, 5_000_000, 1),
            Self::WideBelow => Scenario {
                payload_kind: PayloadKind::Wide,
                ..narrow(100_000, 48_000, 48)
            },
            Self::WideAt => Scenario {
                payload_kind: PayloadKind::Wide,
                ..narrow(100_000, 64_000, 64)
            },
            Self::WideAbove => Scenario {
                payload_kind: PayloadKind::Wide,
                ..narrow(100_000, 80_000, 80)
            },
            Self::StringKeys => Scenario {
                key_kind: KeyKind::String32,
                ..narrow(2_000_000, 2_000_000, 100)
            },
            Self::Presorted => Scenario {
                rhs_order: InputOrder::Sorted,
                ..narrow(5_000_000, 5_000_000, 100)
            },
            Self::Scale250k => narrow(250_000, 250_000, 100),
            Self::Scale1m => narrow(1_000_000, 1_000_000, 100),
            Self::Scale2m => narrow(2_000_000, 2_000_000, 100),
            Self::Scale5m => narrow(5_000_000, 5_000_000, 100),
            Self::Wide520 => wide_memory(520),
            Self::Wide1024 => wide_memory(1_024),
            Self::Wide2048 => wide_memory(2_048),
            Self::Wide4096 => wide_memory(4_096),
            Self::Wide8192 => wide_memory(8_192),
            Self::Wide16384 => wide_memory(16_384),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SmokeNarrow => "smoke-narrow",
            Self::SmallNarrow => "small-narrow",
            Self::RowBelow => "row-below",
            Self::RowAt => "row-at",
            Self::RowAbove => "row-above",
            Self::LargeDense => "large-dense",
            Self::LargeSparse => "large-sparse",
            Self::WideBelow => "wide-below",
            Self::WideAt => "wide-at",
            Self::WideAbove => "wide-above",
            Self::StringKeys => "string-keys",
            Self::Presorted => "presorted",
            Self::Scale250k => "scale250k",
            Self::Scale1m => "scale1m",
            Self::Scale2m => "scale2m",
            Self::Scale5m => "scale5m",
            Self::Wide520 => "wide520",
            Self::Wide1024 => "wide1024",
            Self::Wide2048 => "wide2048",
            Self::Wide4096 => "wide4096",
            Self::Wide8192 => "wide8192",
            Self::Wide16384 => "wide16384",
        }
    }
}

impl StrategyArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Auto => "auto",
            Self::Hash => "hash",
            Self::SortMerge => "sort-merge",
        }
    }
}

fn scenarios(suite: Suite) -> Vec<ScenarioName> {
    const THRESHOLDS: &[ScenarioName] = &[
        ScenarioName::RowBelow,
        ScenarioName::RowAt,
        ScenarioName::RowAbove,
        ScenarioName::WideBelow,
        ScenarioName::WideAt,
        ScenarioName::WideAbove,
    ];
    const CORE: &[ScenarioName] = &[
        ScenarioName::SmallNarrow,
        ScenarioName::LargeDense,
        ScenarioName::LargeSparse,
        ScenarioName::WideAbove,
        ScenarioName::StringKeys,
        ScenarioName::Presorted,
    ];
    const SCALING: &[ScenarioName] = &[
        ScenarioName::Scale250k,
        ScenarioName::Scale1m,
        ScenarioName::Scale2m,
        ScenarioName::Scale5m,
    ];
    const WIDE_MEMORY: &[ScenarioName] = &[
        ScenarioName::Wide520,
        ScenarioName::Wide1024,
        ScenarioName::Wide2048,
        ScenarioName::Wide4096,
    ];
    match suite {
        Suite::Smoke => vec![ScenarioName::SmokeNarrow],
        Suite::Thresholds => THRESHOLDS.to_vec(),
        Suite::Core => CORE.to_vec(),
        Suite::Scaling => SCALING.to_vec(),
        Suite::WideMemory => WIDE_MEMORY.to_vec(),
        Suite::All => {
            let mut all = vec![ScenarioName::SmokeNarrow];
            all.extend_from_slice(THRESHOLDS);
            for scenario in CORE {
                if !all.contains(scenario) {
                    all.push(*scenario);
                }
            }
            all.extend_from_slice(SCALING);
            all
        }
    }
}

#[derive(Clone, Copy)]
enum InputSide {
    Left,
    Right,
}

struct SyntheticReader {
    scenario: Scenario,
    side: InputSide,
    schema: SchemaRef,
    total_rows: u64,
    position: u64,
    permutation_multiplier: u64,
}

impl SyntheticReader {
    fn new(scenario: Scenario, side: InputSide) -> Self {
        let total_rows = match side {
            InputSide::Left => scenario.left_rows,
            InputSide::Right => scenario.rhs_rows,
        };
        Self {
            scenario,
            side,
            schema: scenario_schema(scenario),
            total_rows,
            position: 0,
            permutation_multiplier: permutation_multiplier(total_rows),
        }
    }

    fn input_order(&self) -> InputOrder {
        match self.side {
            InputSide::Left => self.scenario.left_order,
            InputSide::Right => self.scenario.rhs_order,
        }
    }

    fn logical_index(&self, position: u64) -> u64 {
        match self.input_order() {
            InputOrder::Sorted => position,
            InputOrder::Shuffled => {
                if self.total_rows <= 1 {
                    return position;
                }
                let product = u128::from(position) * u128::from(self.permutation_multiplier);
                ((product + u128::from(SHUFFLE_SEED)) % u128::from(self.total_rows)) as u64
            }
        }
    }

    fn key(&self, logical_index: u64) -> u64 {
        match self.side {
            InputSide::Left => logical_index,
            InputSide::Right if logical_index < self.scenario.matched_rows() => logical_index,
            InputSide::Right => self
                .scenario
                .left_rows
                .saturating_add(logical_index - self.scenario.matched_rows()),
        }
    }

    fn make_batch(&self, start: u64, rows: usize) -> Result<RecordBatch, ArrowError> {
        let keys = (start..start + rows as u64)
            .map(|position| self.key(self.logical_index(position)))
            .collect::<Vec<_>>();
        let key_array: ArrayRef = match self.scenario.key_kind {
            KeyKind::UInt64 => Arc::new(UInt64Array::from(keys.clone())),
            KeyKind::String32 => Arc::new(StringArray::from_iter_values(
                keys.iter().map(|key| format!("{key:032x}")),
            )),
        };
        let value_array: ArrayRef = match self.scenario.payload_kind {
            PayloadKind::Narrow => Arc::new(Int64Array::from(
                keys.iter()
                    .map(|key| narrow_value(*key, self.side))
                    .collect::<Vec<_>>(),
            )),
            PayloadKind::Wide => {
                let dimension = WIDE_DIMENSION as usize;
                let value_count = rows.checked_mul(dimension).ok_or_else(|| {
                    ArrowError::InvalidArgumentError(
                        "wide benchmark batch value count overflowed usize".to_string(),
                    )
                })?;
                let mut values = Vec::with_capacity(value_count);
                for key in &keys {
                    let value = wide_value(*key, self.side);
                    values.extend(std::iter::repeat_n(value, dimension));
                }
                Arc::new(FixedSizeListArray::try_new(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    WIDE_DIMENSION,
                    Arc::new(Float32Array::from(values)),
                    None,
                )?)
            }
        };
        RecordBatch::try_new(self.schema.clone(), vec![key_array, value_array])
    }
}

impl Iterator for SyntheticReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.total_rows {
            return None;
        }
        let remaining = self.total_rows - self.position;
        let rows = remaining.min(DEFAULT_BATCH_SIZE as u64) as usize;
        let result = self.make_batch(self.position, rows);
        self.position += rows as u64;
        Some(result)
    }
}

impl RecordBatchReader for SyntheticReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

fn scenario_schema(scenario: Scenario) -> SchemaRef {
    let key_type = match scenario.key_kind {
        KeyKind::UInt64 => DataType::UInt64,
        KeyKind::String32 => DataType::Utf8,
    };
    let value_type = match scenario.payload_kind {
        PayloadKind::Narrow => DataType::Int64,
        PayloadKind::Wide => DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            WIDE_DIMENSION,
        ),
    };
    Arc::new(ArrowSchema::new(vec![
        Field::new(KEY_COLUMN, key_type, false),
        Field::new(VALUE_COLUMN, value_type, false),
    ]))
}

fn permutation_multiplier(total_rows: u64) -> u64 {
    if total_rows <= 1 {
        return 1;
    }
    let mut candidate = (SHUFFLE_SEED % total_rows) | 1;
    while greatest_common_divisor(candidate, total_rows) != 1 {
        candidate = (candidate + 2) % total_rows;
        if candidate == 0 {
            candidate = 1;
        }
    }
    candidate
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn narrow_value(key: u64, side: InputSide) -> i64 {
    let key = i64::try_from(key).expect("benchmark keys fit i64");
    match side {
        InputSide::Left => -key - 1,
        InputSide::Right => key + 1,
    }
}

fn wide_value(key: u64, side: InputSide) -> f32 {
    let value = (key % 1_024) as f32 + 1.0;
    match side {
        InputSide::Left => -value,
        InputSide::Right => value,
    }
}

#[derive(Default)]
struct CollectedMetrics {
    counts: HashMap<String, usize>,
    times: HashMap<String, usize>,
}

impl CollectedMetrics {
    fn record(&mut self, summary: &ExecutionSummaryCounts) {
        for (name, value) in &summary.all_counts {
            *self.counts.entry(name.clone()).or_default() += value;
        }
        for (name, value) in &summary.all_times {
            *self.times.entry(name.clone()).or_default() += value;
        }
    }

    fn count(&self, name: &str) -> usize {
        self.counts.get(name).copied().unwrap_or_default()
    }

    fn selected_strategy(&self) -> &'static str {
        for strategy in ["hash", "sort_merge", "empty"] {
            if self.count(&format!("update_join_strategy_{strategy}")) > 0 {
                return strategy;
            }
        }
        "unknown"
    }
}

#[derive(Serialize)]
struct BenchmarkRecord {
    scenario: String,
    requested_strategy: String,
    selected_strategy: String,
    repetition: usize,
    warmup: bool,
    left_rows: u64,
    rhs_rows: u64,
    matched_rows: u64,
    key_kind: String,
    payload_kind: String,
    target_rhs_estimated_mib: Option<u64>,
    memory_pool_mib: u64,
    execution_batch_size: usize,
    sort_spill_reservation_mib: u64,
    max_temp_gib: u64,
    max_hash_rows: Option<usize>,
    max_hash_mib: Option<u64>,
    elapsed_seconds: f64,
    rows_per_second: f64,
    cpu_seconds: f64,
    baseline_rss_mib: f64,
    peak_rss_mib: f64,
    peak_rss_increase_mib: f64,
    spill_count: usize,
    spilled_rows: usize,
    spilled_bytes: usize,
    rhs_payload_materialization_spill_count: usize,
    rhs_payload_materialization_spilled_rows: usize,
    rhs_payload_materialization_spilled_bytes: usize,
    rhs_sort_spill_count: usize,
    rhs_sort_spilled_rows: usize,
    rhs_sort_spilled_bytes: usize,
    rhs_materialization_spill_count: usize,
    rhs_materialization_spilled_rows: usize,
    rhs_materialization_spilled_bytes: usize,
    join_pipeline_spill_count: usize,
    join_pipeline_spilled_rows: usize,
    join_pipeline_spilled_bytes: usize,
    patch_materialization_spill_count: usize,
    patch_materialization_spilled_rows: usize,
    patch_materialization_spilled_bytes: usize,
    patch_mapping_sort_spill_count: usize,
    patch_mapping_sort_spilled_rows: usize,
    patch_mapping_sort_spilled_bytes: usize,
    patch_mapping_materialization_spill_count: usize,
    patch_mapping_materialization_spilled_rows: usize,
    patch_mapping_materialization_spilled_bytes: usize,
    patch_sort_spill_count: usize,
    patch_sort_spilled_rows: usize,
    patch_sort_spilled_bytes: usize,
    rhs_rows_at_selection: usize,
    rhs_estimated_bytes_at_selection: usize,
    output_bytes: u64,
    checksum: u64,
    status: &'static str,
}

struct RssSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RssSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(current_rss_bytes()));
        let worker_stop = stop.clone();
        let worker_peak = peak.clone();
        let handle = thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                worker_peak.fetch_max(current_rss_bytes(), Ordering::Relaxed);
                thread::sleep(Duration::from_millis(2));
            }
        });
        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    fn stop(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.peak.load(Ordering::Relaxed)
    }
}

fn current_rss_bytes() -> u64 {
    let Ok(statm) = fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    let Some(resident_pages) = statm
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
    else {
        return 0;
    };
    resident_pages.saturating_mul(page_size())
}

#[cfg(unix)]
fn page_size() -> u64 {
    // SAFETY: sysconf has no memory-safety preconditions.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(size).unwrap_or(4_096)
}

#[cfg(not(unix))]
fn page_size() -> u64 {
    4_096
}

#[cfg(unix)]
fn process_cpu_seconds() -> f64 {
    // SAFETY: getrusage initializes the provided rusage value on success.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return 0.0;
        }
        usage.ru_utime.tv_sec as f64
            + usage.ru_utime.tv_usec as f64 / 1_000_000.0
            + usage.ru_stime.tv_sec as f64
            + usage.ru_stime.tv_usec as f64 / 1_000_000.0
    }
}

#[cfg(not(unix))]
fn process_cpu_seconds() -> f64 {
    0.0
}

async fn run_worker(args: &Args) -> BenchResult<BenchmarkRecord> {
    let scenario = args
        .scenario
        .ok_or("worker mode requires an explicit --scenario")?
        .config();
    if args.strategy == StrategyArg::All {
        return Err("worker mode requires one concrete --strategy".into());
    }
    validate_threshold_overrides(args)?;
    validate_execution_overrides(args)?;

    let directory = TempStrDir::default();
    let dataset = Dataset::write(
        SyntheticReader::new(scenario, InputSide::Left),
        directory.as_ref(),
        Some(WriteParams {
            max_rows_per_file: usize::try_from(scenario.left_rows)?,
            ..Default::default()
        }),
    )
    .await?;
    if dataset.get_fragments().len() != 1 {
        return Err(format!(
            "benchmark expected one fragment, created {}",
            dataset.get_fragments().len()
        )
        .into());
    }
    let baseline_dataset_bytes = directory_size(Path::new(directory.as_ref()))?;
    let mut fragment = dataset
        .get_fragment(0)
        .ok_or("benchmark dataset has no fragment 0")?;
    let metrics = Arc::new(Mutex::new(CollectedMetrics::default()));
    let callback_metrics = metrics.clone();
    let execution_options = LanceExecutionOptions {
        use_spilling: true,
        mem_pool_size: Some(
            args.memory_pool_mib
                .checked_mul(MIB)
                .ok_or("memory_pool_mib exceeds the representable byte count")?,
        ),
        max_temp_directory_size: Some(
            args.max_temp_gib
                .checked_mul(GIB)
                .ok_or("max_temp_gib exceeds the representable byte count")?,
        ),
        target_partition: Some(1),
        execution_stats_callback: Some(Arc::new(move |summary| {
            callback_metrics.lock().unwrap().record(summary);
        })),
        skip_logging: true,
        ..Default::default()
    };
    let join_options = benchmark_join_options(args)?;
    let sort_spill_reservation_bytes = args
        .sort_spill_reservation_mib
        .checked_mul(MIB)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("sort_spill_reservation_mib exceeds the representable byte count")?;
    let right_reader = SyntheticReader::new(scenario, InputSide::Right);

    let baseline_rss = current_rss_bytes();
    let rss_sampler = RssSampler::start();
    let cpu_before = process_cpu_seconds();
    let started = Instant::now();
    let update_result = update_columns(
        &mut fragment,
        right_reader,
        KEY_COLUMN,
        KEY_COLUMN,
        UpdateJoinBenchmarkOptions {
            join_options,
            execution_options,
            execution_batch_size: Some(args.execution_batch_size),
            sort_spill_reservation_bytes: Some(sort_spill_reservation_bytes),
        },
    )
    .await;
    let elapsed = started.elapsed();
    let cpu_seconds = process_cpu_seconds() - cpu_before;
    let peak_rss = rss_sampler.stop();
    let update_result = update_result?;
    let output_bytes =
        directory_size(Path::new(directory.as_ref()))?.saturating_sub(baseline_dataset_bytes);
    let matched_rows = update_result.matched_offsets.len();
    if matched_rows != scenario.matched_rows() {
        return Err(format!(
            "matched row count was {matched_rows}, expected {}",
            scenario.matched_rows()
        )
        .into());
    }
    let checksum = validate_output(Arc::new(dataset), &update_result, scenario).await?;
    let metrics = metrics.lock().unwrap();
    let elapsed_seconds = elapsed.as_secs_f64();

    Ok(BenchmarkRecord {
        scenario: scenario.name.as_str().to_string(),
        requested_strategy: requested_strategy_name(args).to_string(),
        selected_strategy: metrics.selected_strategy().to_string(),
        repetition: args.repetition,
        warmup: args.warmup,
        left_rows: scenario.left_rows,
        rhs_rows: scenario.rhs_rows,
        matched_rows,
        key_kind: format!("{:?}", scenario.key_kind),
        payload_kind: format!("{:?}", scenario.payload_kind),
        target_rhs_estimated_mib: scenario.target_rhs_estimated_mib(),
        memory_pool_mib: args.memory_pool_mib,
        execution_batch_size: args.execution_batch_size,
        sort_spill_reservation_mib: args.sort_spill_reservation_mib,
        max_temp_gib: args.max_temp_gib,
        max_hash_rows: args.max_hash_rows,
        max_hash_mib: args.max_hash_mib,
        elapsed_seconds,
        rows_per_second: scenario.left_rows as f64 / elapsed_seconds,
        cpu_seconds,
        baseline_rss_mib: bytes_to_mib(baseline_rss),
        peak_rss_mib: bytes_to_mib(peak_rss),
        peak_rss_increase_mib: bytes_to_mib(peak_rss.saturating_sub(baseline_rss)),
        spill_count: metrics.count("spill_count"),
        spilled_rows: metrics.count("spilled_rows"),
        spilled_bytes: metrics.count("spilled_bytes"),
        rhs_payload_materialization_spill_count: metrics
            .count("update_join_rhs_payload_materialization_spill_count"),
        rhs_payload_materialization_spilled_rows: metrics
            .count("update_join_rhs_payload_materialization_spilled_rows"),
        rhs_payload_materialization_spilled_bytes: metrics
            .count("update_join_rhs_payload_materialization_spilled_bytes"),
        rhs_sort_spill_count: metrics.count("update_join_rhs_sort_spill_count"),
        rhs_sort_spilled_rows: metrics.count("update_join_rhs_sort_spilled_rows"),
        rhs_sort_spilled_bytes: metrics.count("update_join_rhs_sort_spilled_bytes"),
        rhs_materialization_spill_count: metrics
            .count("update_join_rhs_materialization_spill_count"),
        rhs_materialization_spilled_rows: metrics
            .count("update_join_rhs_materialization_spilled_rows"),
        rhs_materialization_spilled_bytes: metrics
            .count("update_join_rhs_materialization_spilled_bytes"),
        join_pipeline_spill_count: metrics.count("update_join_join_pipeline_spill_count"),
        join_pipeline_spilled_rows: metrics.count("update_join_join_pipeline_spilled_rows"),
        join_pipeline_spilled_bytes: metrics.count("update_join_join_pipeline_spilled_bytes"),
        patch_materialization_spill_count: metrics
            .count("update_join_patch_materialization_spill_count"),
        patch_materialization_spilled_rows: metrics
            .count("update_join_patch_materialization_spilled_rows"),
        patch_materialization_spilled_bytes: metrics
            .count("update_join_patch_materialization_spilled_bytes"),
        patch_mapping_sort_spill_count: metrics.count("update_join_patch_mapping_sort_spill_count"),
        patch_mapping_sort_spilled_rows: metrics
            .count("update_join_patch_mapping_sort_spilled_rows"),
        patch_mapping_sort_spilled_bytes: metrics
            .count("update_join_patch_mapping_sort_spilled_bytes"),
        patch_mapping_materialization_spill_count: metrics
            .count("update_join_patch_mapping_materialization_spill_count"),
        patch_mapping_materialization_spilled_rows: metrics
            .count("update_join_patch_mapping_materialization_spilled_rows"),
        patch_mapping_materialization_spilled_bytes: metrics
            .count("update_join_patch_mapping_materialization_spilled_bytes"),
        patch_sort_spill_count: metrics.count("update_join_patch_sort_spill_count"),
        patch_sort_spilled_rows: metrics.count("update_join_patch_sort_spilled_rows"),
        patch_sort_spilled_bytes: metrics.count("update_join_patch_sort_spilled_bytes"),
        rhs_rows_at_selection: metrics.count("update_join_rhs_rows_at_selection"),
        rhs_estimated_bytes_at_selection: metrics
            .count("update_join_rhs_estimated_bytes_at_selection"),
        output_bytes,
        checksum,
        status: "ok",
    })
}

fn benchmark_join_options(args: &Args) -> BenchResult<UpdateJoinOptions> {
    match args.strategy {
        StrategyArg::All => Err("worker strategy cannot be all".into()),
        StrategyArg::Hash => {
            Ok(UpdateJoinOptions::default().with_strategy(UpdateJoinStrategy::Hash))
        }
        StrategyArg::SortMerge => {
            Ok(UpdateJoinOptions::default().with_strategy(UpdateJoinStrategy::SortMerge))
        }
        StrategyArg::Auto => {
            match (args.max_hash_rows, args.max_hash_mib) {
                (None, None) => Ok(UpdateJoinOptions::default()),
                (Some(max_hash_rows), Some(max_hash_mib)) => {
                    let max_hash_bytes = max_hash_mib
                        .checked_mul(MIB)
                        .and_then(|bytes| usize::try_from(bytes).ok())
                        .ok_or("max_hash_mib exceeds the representable byte count")?;
                    Ok(UpdateJoinOptions::default()
                        .with_hash_thresholds(max_hash_rows, max_hash_bytes))
                }
                _ => Err("max_hash_rows and max_hash_mib must be supplied together".into()),
            }
        }
    }
}

fn requested_strategy_name(args: &Args) -> &'static str {
    if args.strategy == StrategyArg::Auto && args.max_hash_rows.is_some() {
        "thresholds"
    } else {
        args.strategy.as_str()
    }
}

fn validate_threshold_overrides(args: &Args) -> BenchResult<()> {
    match (args.max_hash_rows, args.max_hash_mib) {
        (None, None) | (Some(_), Some(_)) => Ok(()),
        _ => Err("max_hash_rows and max_hash_mib must be supplied together".into()),
    }
}

fn validate_execution_overrides(args: &Args) -> BenchResult<()> {
    if args.execution_batch_size == 0 {
        return Err("execution_batch_size must be greater than zero".into());
    }
    if args.sort_spill_reservation_mib == 0 {
        return Err("sort_spill_reservation_mib must be greater than zero".into());
    }
    if args.sort_spill_reservation_mib >= args.memory_pool_mib {
        return Err(format!(
            "sort_spill_reservation_mib must be smaller than memory_pool_mib: reservation={}, pool={}",
            args.sort_spill_reservation_mib, args.memory_pool_mib
        )
        .into());
    }
    Ok(())
}

async fn validate_output(
    dataset: Arc<Dataset>,
    update_result: &FragmentUpdateColumnsResult,
    scenario: Scenario,
) -> BenchResult<u64> {
    let updated_fragment = FileFragment::new(dataset, update_result.fragment.clone());
    let mut stream = updated_fragment.scan().try_into_stream().await?;
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    let mut rows = 0_u64;
    while let Some(batch) = stream.try_next().await? {
        let keys = batch
            .column_by_name(KEY_COLUMN)
            .ok_or("updated batch has no key column")?;
        let values = batch
            .column_by_name(VALUE_COLUMN)
            .ok_or("updated batch has no value column")?;
        for row in 0..batch.num_rows() {
            let key = match scenario.key_kind {
                KeyKind::UInt64 => keys
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or("updated key column is not UInt64")?
                    .value(row),
                KeyKind::String32 => {
                    let value = keys
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("updated key column is not Utf8")?
                        .value(row);
                    u64::from_str_radix(value, 16)?
                }
            };
            let value_bits = match scenario.payload_kind {
                PayloadKind::Narrow => values
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or("updated value column is not Int64")?
                    .value(row) as u64,
                PayloadKind::Wide => {
                    let lists = values
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .ok_or("updated value column is not FixedSizeList")?;
                    let child = lists
                        .values()
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or("updated wide value child is not Float32")?;
                    child.value(row * WIDE_DIMENSION as usize).to_bits() as u64
                }
            };
            checksum = extend_checksum(checksum, key, value_bits);
            rows += 1;
        }
    }
    if rows != scenario.left_rows {
        return Err(format!(
            "updated output contained {rows} rows, expected {}",
            scenario.left_rows
        )
        .into());
    }
    let expected = expected_checksum(scenario);
    if checksum != expected {
        return Err(
            format!("updated output checksum was {checksum:#x}, expected {expected:#x}").into(),
        );
    }
    Ok(checksum)
}

fn expected_checksum(scenario: Scenario) -> u64 {
    let multiplier = permutation_multiplier(scenario.left_rows);
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for position in 0..scenario.left_rows {
        let key = match scenario.left_order {
            InputOrder::Sorted => position,
            InputOrder::Shuffled if scenario.left_rows <= 1 => position,
            InputOrder::Shuffled => {
                let product = u128::from(position) * u128::from(multiplier);
                ((product + u128::from(SHUFFLE_SEED)) % u128::from(scenario.left_rows)) as u64
            }
        };
        let side = if key < scenario.matched_rows() {
            InputSide::Right
        } else {
            InputSide::Left
        };
        let value_bits = match scenario.payload_kind {
            PayloadKind::Narrow => narrow_value(key, side) as u64,
            PayloadKind::Wide => wide_value(key, side).to_bits() as u64,
        };
        checksum = extend_checksum(checksum, key, value_bits);
    }
    checksum
}

fn extend_checksum(checksum: u64, key: u64, value_bits: u64) -> u64 {
    checksum
        .wrapping_mul(0x0000_0100_0000_01B3)
        .wrapping_add(key.rotate_left(17) ^ value_bits)
}

fn directory_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0_u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total = total.saturating_add(directory_size(&entry.path())?);
        } else {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / MIB as f64
}

fn concrete_strategies(requested: StrategyArg, scenario: Scenario) -> Vec<StrategyArg> {
    if requested != StrategyArg::All {
        return vec![requested];
    }
    let mut strategies = vec![StrategyArg::Hash, StrategyArg::SortMerge];
    if scenario.is_threshold() || matches!(scenario.name, ScenarioName::SmokeNarrow) {
        strategies.push(StrategyArg::Auto);
    }
    strategies
}

fn run_parent(args: &Args) -> BenchResult<()> {
    validate_threshold_overrides(args)?;
    validate_execution_overrides(args)?;
    let selected_scenarios = args
        .scenario
        .map(|scenario| vec![scenario])
        .unwrap_or_else(|| scenarios(args.suite));
    let executable = std::env::current_exe()?;

    for scenario_name in selected_scenarios {
        let scenario = scenario_name.config();
        for run_index in 0..args.warmups + args.repetitions {
            let is_warmup = run_index < args.warmups;
            let repetition = run_index.saturating_sub(args.warmups);
            let mut strategies = concrete_strategies(args.strategy, scenario);
            if run_index % 2 == 1 {
                strategies.reverse();
            }
            for strategy in strategies {
                let mut command = Command::new(&executable);
                command
                    .arg("--worker")
                    .arg("--scenario")
                    .arg(scenario_name.as_str())
                    .arg("--strategy")
                    .arg(strategy.as_str())
                    .arg("--repetition")
                    .arg(repetition.to_string())
                    .arg("--memory-pool-mib")
                    .arg(args.memory_pool_mib.to_string())
                    .arg("--execution-batch-size")
                    .arg(args.execution_batch_size.to_string())
                    .arg("--sort-spill-reservation-mib")
                    .arg(args.sort_spill_reservation_mib.to_string())
                    .arg("--max-temp-gib")
                    .arg(args.max_temp_gib.to_string());
                if is_warmup {
                    command.arg("--warmup");
                }
                if let (Some(max_hash_rows), Some(max_hash_mib)) =
                    (args.max_hash_rows, args.max_hash_mib)
                    && strategy == StrategyArg::Auto
                {
                    command
                        .arg("--max-hash-rows")
                        .arg(max_hash_rows.to_string())
                        .arg("--max-hash-mib")
                        .arg(max_hash_mib.to_string());
                }
                let output = command.output()?;
                if !output.status.success() {
                    return Err(format!(
                        "worker failed for scenario {} strategy {} with status {}: {}",
                        scenario_name.as_str(),
                        strategy.as_str(),
                        output.status,
                        String::from_utf8_lossy(&output.stderr)
                    )
                    .into());
                }
                if !is_warmup {
                    print!("{}", String::from_utf8(output.stdout)?);
                }
            }
        }
    }
    Ok(())
}

fn main() -> BenchResult<()> {
    let args = Args::parse();
    if args.worker {
        let runtime = tokio::runtime::Runtime::new()?;
        let record = runtime.block_on(run_worker(&args))?;
        println!("{}", serde_json::to_string(&record)?);
        Ok(())
    } else {
        run_parent(&args)
    }
}
