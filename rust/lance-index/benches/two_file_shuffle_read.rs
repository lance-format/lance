// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reproducible read benchmark for the two-file IVF shuffle format.
//!
//! The fixture is deliberately generated once, before Criterion starts timing,
//! and is then reopened for every end-to-end sample.  Each input `RecordBatch` is forced to
//! become one flush group, so both scenarios contain exactly 20 physical groups.
//! The payload has the same column topology as a 5-bit RQ build (row ID, binary
//! and extra codes, and five floating-point factors), scaled to 256 dimensions
//! so the benchmark remains practical on a developer laptop.
//!
//! Run both the benchmark-local pre-change reader and the current reader against
//! one persistent fixture:
//!
//! ```text
//! LANCE_SHUFFLE_BENCH_FIXTURE_ROOT=/tmp/lance-two-file-fixture \
//! cargo bench --profile release-with-debug -p lance-index \
//!   --bench two_file_shuffle_read
//! ```
//!
//! The optional fixture root makes later invocations reopen the exact same files
//! and manifest. Without it, each process regenerates byte-equivalent
//! deterministic fixtures in temporary directories.
//! A benchmark-local copy of the pre-change on-demand offsets reader runs next
//! to the current reader, so every invocation also provides a same-process,
//! same-file baseline/current comparison.
//!
//! Criterion reports separate reopen, read-only, and reopen-plus-read timings;
//! the latter two report rows/s.  Before each scenario it also prints one
//! cache-hot diagnostic pass containing separate init/read/total wall time,
//! process CPU time, peak RSS, scheduler IOPS, scheduler bytes read, and the
//! number of logical data ranges requested.  Scheduler counters are split by
//! phase but aggregate data and offsets files; exact per-file calls require the
//! scheduler's per-file trace events.  For kernel syscall counts on Linux, wrap
//! either command with `strace -f -c -e pread64`.
//! `LANCE_SHUFFLE_BENCH_DISTRIBUTION` (`uniform` or `hotspot`) and
//! `LANCE_SHUFFLE_BENCH_IMPLEMENTATION` (`baseline` or `current`) can isolate a
//! scenario in a fresh process so peak RSS is comparable.

use std::hint::black_box;
use std::io::Write;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::{array::AsArray, compute::concat_batches, datatypes::UInt64Type};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array, UInt32Array, UInt64Array,
};
use arrow_schema::Schema;
use async_trait::async_trait;
use criterion::{Criterion, Throughput};
use futures::{StreamExt, TryStreamExt, stream};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempDir;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{Error, ROW_ID};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_index::vector::PART_ID_COLUMN;
use lance_index::vector::bq::ex_dot::blocked_ex_code_bytes;
use lance_index::vector::bq::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN};
use lance_index::vector::bq::transform::{
    ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
    SCALE_FACTORS_COLUMN,
};
use lance_index::vector::bq::{rabit_binary_code_bytes, rabit_ex_bits};
use lance_index::vector::v3::shuffle_bench::{
    TwoFileShuffleFixtureManifest, open_two_file_shuffle_fixture,
};
use lance_index::vector::v3::shuffler::{
    DEFAULT_PARTITION_WINDOW_BYTES, ShuffleReader, Shuffler, TwoFileShuffler,
};
use lance_io::ReadBatchParams;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::scheduler::{bytes_read_counter, iops_counter};
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;

const NUM_FLUSH_GROUPS: usize = 20;
const NUM_PARTITIONS: usize = 4_096;
const ROWS_PER_PARTITION: usize = 64;
const RQ_DIMENSION: usize = 256;
const RQ_NUM_BITS: u8 = 5;
const HOTSPOT_TARGET_CV: f64 = 3.6;

#[derive(Clone, Copy, Debug)]
enum Distribution {
    Uniform,
    Hotspot,
}

#[derive(Clone, Copy, Debug)]
enum ReaderImplementation {
    Baseline,
    Current,
}

impl ReaderImplementation {
    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline_on_demand_offsets",
            Self::Current => "current",
        }
    }
}

/// The pre-change two-file reader, kept local to the benchmark so baseline and
/// current read the exact same files in the same process.
struct BaselineTwoFileShuffleReader {
    _scheduler: Arc<ScanScheduler>,
    file_reader: FileReader,
    offsets_reader: FileReader,
    num_partitions: usize,
    num_flush_groups: u64,
    partition_counts: Vec<u64>,
    total_loss: f64,
}

impl BaselineTwoFileShuffleReader {
    async fn try_new(
        output_dir: Path,
        manifest: &TwoFileShuffleFixtureManifest,
    ) -> lance_core::Result<Arc<dyn ShuffleReader>> {
        let object_store = Arc::new(ObjectStore::local());
        let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
        let scheduler = ScanScheduler::new(object_store, scheduler_config);

        let data_path = output_dir.clone().join("shuffle_data.lance");
        let file_reader = FileReader::try_open(
            scheduler
                .open_file(&data_path, &CachedFileSize::unknown())
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;

        let offsets_path = output_dir.join("shuffle_offsets.lance");
        let offsets_reader = FileReader::try_open(
            scheduler
                .open_file(&offsets_path, &CachedFileSize::unknown())
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;

        Ok(Arc::new(Self {
            _scheduler: scheduler,
            file_reader,
            offsets_reader,
            num_partitions: manifest.num_partitions,
            num_flush_groups: manifest.num_flush_groups,
            partition_counts: manifest.partition_counts.clone(),
            total_loss: manifest.total_loss,
        }))
    }

    async fn partition_ranges(&self, partition_id: usize) -> lance_core::Result<Vec<Range<u64>>> {
        let mut positions = Vec::with_capacity(self.num_flush_groups as usize * 2);
        for group in 0..self.num_flush_groups {
            let end_position = u32::try_from(group as usize * self.num_partitions + partition_id)
                .map_err(|_| {
                Error::invalid_input(
                    "There are more than 2^32 partition offsets in the spill file. Need to support 64-bit take",
                )
            })?;
            if end_position != 0 {
                positions.push(end_position - 1);
            }
            positions.push(end_position);
        }

        let num_positions = positions.len() as u32;
        let positions = UInt32Array::from(positions);
        let offsets_stream = self
            .offsets_reader
            .read_stream(
                ReadBatchParams::Indices(positions),
                num_positions,
                1,
                FilterExpression::no_filter(),
            )
            .await?;
        let schema = offsets_stream.schema().clone();
        let offsets = offsets_stream.try_collect::<Vec<_>>().await?;
        let offsets = if offsets.len() == 1 {
            offsets.into_iter().next().expect("one offsets batch")
        } else {
            concat_batches(&schema, &offsets)?
        };
        let offsets = offsets.column(0).as_primitive::<UInt64Type>();
        let mut offsets_iter = offsets.values().iter().copied();

        let mut ranges = Vec::with_capacity(self.num_flush_groups as usize);
        for group in 0..self.num_flush_groups {
            if group == 0 && partition_id == 0 {
                ranges.push(0..offsets_iter.next().expect("partition end offset"));
            } else {
                ranges.push(
                    offsets_iter.next().expect("partition start offset")
                        ..offsets_iter.next().expect("partition end offset"),
                );
            }
        }
        Ok(ranges)
    }
}

#[async_trait]
impl ShuffleReader for BaselineTwoFileShuffleReader {
    async fn read_partition(
        &self,
        partition_id: usize,
    ) -> lance_core::Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>> {
        if partition_id >= self.num_partitions || self.partition_counts[partition_id] == 0 {
            return Ok(None);
        }

        let ranges = self.partition_ranges(partition_id).await?;
        let schema: Schema = self.file_reader.schema().as_ref().into();
        let stream = self
            .file_reader
            .read_stream(
                ReadBatchParams::Ranges(ranges.into()),
                u32::MAX,
                16,
                FilterExpression::no_filter(),
            )
            .await?;
        Ok(Some(Box::new(RecordBatchStreamAdapter::new(
            Arc::new(schema),
            stream,
        ))))
    }

    fn partition_size(&self, partition_id: usize) -> lance_core::Result<usize> {
        Ok(self
            .partition_counts
            .get(partition_id)
            .copied()
            .unwrap_or(0) as usize)
    }

    fn total_loss(&self) -> Option<f64> {
        Some(self.total_loss)
    }
}

impl Distribution {
    fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Hotspot => "hotspot_cv_3_6",
        }
    }
}

struct Fixture {
    _temporary_directory: Option<TempDir>,
    output_dir: Path,
    manifest_path: PathBuf,
    manifest: TwoFileShuffleFixtureManifest,
    baseline_reader: Arc<dyn ShuffleReader>,
    current_reader: Arc<dyn ShuffleReader>,
    total_rows: u64,
    coefficient_of_variation: f64,
}

fn partition_counts(distribution: Distribution) -> Vec<usize> {
    match distribution {
        Distribution::Uniform => vec![ROWS_PER_PARTITION; NUM_PARTITIONS],
        Distribution::Hotspot => {
            let total_rows = NUM_PARTITIONS * ROWS_PER_PARTITION;
            let hotspot_rows = (ROWS_PER_PARTITION as f64
                * (1.0 + HOTSPOT_TARGET_CV * ((NUM_PARTITIONS - 1) as f64).sqrt()))
            .round() as usize;
            let other_rows = total_rows - hotspot_rows;
            let base = other_rows / (NUM_PARTITIONS - 1);
            let remainder = other_rows % (NUM_PARTITIONS - 1);

            let hotspot_partition = NUM_PARTITIONS / 2;
            let mut counts = Vec::with_capacity(NUM_PARTITIONS);
            for partition_id in 0..NUM_PARTITIONS {
                if partition_id == hotspot_partition {
                    counts.push(hotspot_rows);
                } else {
                    let non_hotspot_index = if partition_id < hotspot_partition {
                        partition_id
                    } else {
                        partition_id - 1
                    };
                    counts.push(base + usize::from(non_hotspot_index < remainder));
                }
            }
            counts
        }
    }
}

fn coefficient_of_variation(counts: &[usize]) -> f64 {
    let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    let variance = counts
        .iter()
        .map(|&count| {
            let delta = count as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / counts.len() as f64;
    variance.sqrt() / mean
}

fn rows_in_flush_group(partition_rows: usize, partition_id: usize, group: usize) -> usize {
    let base = partition_rows / NUM_FLUSH_GROUPS;
    let remainder = partition_rows % NUM_FLUSH_GROUPS;
    base + usize::from((group + partition_id * 7) % NUM_FLUSH_GROUPS < remainder)
}

fn make_rq5_like_batch(counts: &[usize], group: usize, next_row_id: &mut u64) -> RecordBatch {
    let num_rows = counts
        .iter()
        .enumerate()
        .map(|(partition_id, &rows)| rows_in_flush_group(rows, partition_id, group))
        .sum::<usize>();
    let mut partition_ids = Vec::with_capacity(num_rows);
    let mut row_ids = Vec::with_capacity(num_rows);

    // 4051 is coprime to 4096, so every group visits each partition exactly
    // once but starts with a different deterministic, non-sorted order.
    for slot in 0..NUM_PARTITIONS {
        let partition_id = (slot * 4_051 + group * 997) % NUM_PARTITIONS;
        let group_rows = rows_in_flush_group(counts[partition_id], partition_id, group);
        for _ in 0..group_rows {
            partition_ids.push(partition_id as u32);
            row_ids.push(*next_row_id);
            *next_row_id += 1;
        }
    }

    let binary_code_bytes = rabit_binary_code_bytes(RQ_DIMENSION);
    let ex_bits = rabit_ex_bits(RQ_NUM_BITS).expect("RQ5 must be a valid configuration");
    let ex_code_bytes = blocked_ex_code_bytes(RQ_DIMENSION, ex_bits);
    let make_codes = |width: usize, salt: u8| {
        let values = (0..num_rows * width)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(salt))
            .collect::<Vec<_>>();
        Arc::new(
            FixedSizeListArray::try_new_from_values(UInt8Array::from(values), width as i32)
                .expect("valid fixed-size RQ code array"),
        ) as ArrayRef
    };
    let make_factors = |salt: f32| {
        Arc::new(Float32Array::from_iter_values(
            (0..num_rows).map(|row| salt + (row % 257) as f32 / 257.0),
        )) as ArrayRef
    };

    RecordBatch::try_from_iter(vec![
        (
            PART_ID_COLUMN,
            Arc::new(UInt32Array::from(partition_ids)) as ArrayRef,
        ),
        (ROW_ID, Arc::new(UInt64Array::from(row_ids)) as ArrayRef),
        (RABIT_CODE_COLUMN, make_codes(binary_code_bytes, 11)),
        (ADD_FACTORS_COLUMN, make_factors(1.0)),
        (SCALE_FACTORS_COLUMN, make_factors(2.0)),
        (ERROR_FACTORS_COLUMN, make_factors(3.0)),
        (RABIT_BLOCKED_EX_CODE_COLUMN, make_codes(ex_code_bytes, 19)),
        (EX_ADD_FACTORS_COLUMN, make_factors(4.0)),
        (EX_SCALE_FACTORS_COLUMN, make_factors(5.0)),
    ])
    .expect("all deterministic fixture columns have equal length")
}

fn batches_to_stream(batches: Vec<RecordBatch>) -> Box<dyn RecordBatchStream + Unpin + 'static> {
    let schema = batches
        .first()
        .expect("the fixture always has 20 flush groups")
        .schema();
    Box::new(RecordBatchStreamAdapter::new(
        schema,
        stream::iter(batches.into_iter().map(Ok)),
    ))
}

async fn build_fixture(distribution: Distribution) -> Fixture {
    let counts = partition_counts(distribution);
    let coefficient_of_variation = coefficient_of_variation(&counts);
    match distribution {
        Distribution::Uniform => assert_eq!(coefficient_of_variation, 0.0),
        Distribution::Hotspot => assert!(
            (coefficient_of_variation - HOTSPOT_TARGET_CV).abs() < 0.01,
            "hotspot fixture CV was {coefficient_of_variation}"
        ),
    }

    let total_rows = counts.iter().sum::<usize>() as u64;
    let manifest = TwoFileShuffleFixtureManifest {
        num_partitions: NUM_PARTITIONS,
        num_flush_groups: NUM_FLUSH_GROUPS as u64,
        partition_counts: counts.iter().map(|&count| count as u64).collect(),
        total_loss: 0.0,
    };

    let (temporary_directory, fixture_path) =
        if let Some(root) = std::env::var_os("LANCE_SHUFFLE_BENCH_FIXTURE_ROOT") {
            let fixture_path = PathBuf::from(root).join(distribution.name());
            std::fs::create_dir_all(&fixture_path).expect("create persistent shuffle fixture root");
            (None, fixture_path)
        } else {
            let directory = TempDir::default();
            let fixture_path = directory.std_path().to_owned();
            (Some(directory), fixture_path)
        };
    let output_dir = Path::from_filesystem_path(&fixture_path)
        .expect("shuffle fixture path must be a valid object-store path");
    let manifest_path = fixture_path.join("shuffle_manifest.json");

    if manifest_path.exists() {
        let stored: TwoFileShuffleFixtureManifest = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read existing shuffle fixture manifest"),
        )
        .expect("parse existing shuffle fixture manifest");
        assert_eq!(
            stored.num_partitions, manifest.num_partitions,
            "existing fixture has a different partition count"
        );
        assert_eq!(
            stored.num_flush_groups, manifest.num_flush_groups,
            "existing fixture has a different flush-group count"
        );
        assert_eq!(
            stored.partition_counts, manifest.partition_counts,
            "existing fixture has a different partition distribution"
        );
    } else {
        let mut next_row_id = 0;
        let batches = (0..NUM_FLUSH_GROUPS)
            .map(|group| make_rq5_like_batch(&counts, group, &mut next_row_id))
            .collect::<Vec<_>>();
        assert_eq!(next_row_id, total_rows);

        let shuffler = TwoFileShuffler::new(output_dir.clone(), NUM_PARTITIONS);
        let initial_reader = shuffler
            .shuffle(batches_to_stream(batches))
            .await
            .expect("write and reopen deterministic two-file shuffle fixture");
        drop(initial_reader);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize deterministic shuffle manifest"),
        )
        .expect("write deterministic shuffle manifest");
    }

    let current_reader = open_two_file_shuffle_fixture(output_dir.clone(), &manifest)
        .await
        .expect("reopen frozen two-file shuffle fixture");
    let baseline_reader = BaselineTwoFileShuffleReader::try_new(output_dir.clone(), &manifest)
        .await
        .expect("reopen frozen fixture with baseline reader");

    Fixture {
        _temporary_directory: temporary_directory,
        output_dir,
        manifest_path,
        manifest,
        baseline_reader,
        current_reader: Arc::from(current_reader),
        total_rows,
        coefficient_of_variation,
    }
}

async fn reopen_fixture(
    fixture: &Fixture,
    implementation: ReaderImplementation,
) -> Arc<dyn ShuffleReader> {
    let manifest_bytes =
        std::fs::read(&fixture.manifest_path).expect("read frozen shuffle fixture manifest");
    let manifest: TwoFileShuffleFixtureManifest =
        serde_json::from_slice(&manifest_bytes).expect("parse frozen shuffle fixture manifest");
    assert_eq!(manifest.num_partitions, fixture.manifest.num_partitions);
    match implementation {
        ReaderImplementation::Baseline => {
            BaselineTwoFileShuffleReader::try_new(fixture.output_dir.clone(), &manifest)
                .await
                .expect("reopen fixture with baseline reader")
        }
        ReaderImplementation::Current => Arc::from(
            open_two_file_shuffle_fixture(fixture.output_dir.clone(), &manifest)
                .await
                .expect("reopen fixture with current reader"),
        ),
    }
}

struct ReadResult {
    rows: u64,
    windows: usize,
}

async fn read_all_partitions_baseline(
    reader: Arc<dyn ShuffleReader>,
    concurrency: usize,
) -> lance_core::Result<ReadResult> {
    let rows = stream::iter(0..NUM_PARTITIONS)
        .map(|partition_id| {
            let reader = reader.clone();
            async move {
                let Some(mut batches) = reader.read_partition(partition_id).await? else {
                    return Ok::<_, lance_core::Error>(0u64);
                };
                let mut rows = 0u64;
                while let Some(batch) = batches.try_next().await? {
                    rows += batch.num_rows() as u64;
                    black_box(batch);
                }
                Ok(rows)
            }
        })
        .buffered(concurrency)
        .try_fold(0u64, |total, rows| async move { Ok(total + rows) })
        .await?;
    Ok(ReadResult {
        rows,
        windows: NUM_PARTITIONS,
    })
}

async fn read_all_partitions_current(
    reader: Arc<dyn ShuffleReader>,
) -> lance_core::Result<ReadResult> {
    let mut rows = 0u64;
    let mut windows = 0usize;
    let mut next_partition_id = 0usize;
    while next_partition_id < NUM_PARTITIONS {
        let window = reader
            .read_partition_window(next_partition_id, DEFAULT_PARTITION_WINDOW_BYTES)
            .await?;
        assert_eq!(window.partition_range.start, next_partition_id);
        assert!(window.partition_range.end > next_partition_id);
        assert!(window.partition_range.end <= NUM_PARTITIONS);
        assert_eq!(window.partition_range.len(), window.partitions.len());
        next_partition_id = window.partition_range.end;
        windows += 1;

        for partition in window.partitions {
            if let Some(mut batches) = partition.data {
                while let Some(batch) = batches.try_next().await? {
                    rows += batch.num_rows() as u64;
                    black_box(batch);
                }
            }
        }
    }
    Ok(ReadResult { rows, windows })
}

async fn read_all_partitions(
    reader: Arc<dyn ShuffleReader>,
    implementation: ReaderImplementation,
    concurrency: usize,
) -> lance_core::Result<ReadResult> {
    match implementation {
        ReaderImplementation::Baseline => read_all_partitions_baseline(reader, concurrency).await,
        ReaderImplementation::Current => read_all_partitions_current(reader).await,
    }
}

#[cfg(unix)]
fn process_resources() -> (f64, u64) {
    // SAFETY: getrusage initializes the provided rusage value and does not
    // retain its pointer.  RUSAGE_SELF is valid on all Unix targets.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return (0.0, 0);
        }
        let user_seconds =
            usage.ru_utime.tv_sec as f64 + usage.ru_utime.tv_usec as f64 / 1_000_000.0;
        let system_seconds =
            usage.ru_stime.tv_sec as f64 + usage.ru_stime.tv_usec as f64 / 1_000_000.0;
        #[cfg(target_os = "macos")]
        let peak_rss_bytes = usage.ru_maxrss as u64;
        #[cfg(not(target_os = "macos"))]
        let peak_rss_bytes = usage.ru_maxrss as u64 * 1024;
        (user_seconds + system_seconds, peak_rss_bytes)
    }
}

#[cfg(not(unix))]
fn process_resources() -> (f64, u64) {
    (0.0, 0)
}

async fn print_diagnostic(
    distribution: Distribution,
    implementation: ReaderImplementation,
    fixture: &Fixture,
    concurrency: usize,
) {
    let init_iops_before = iops_counter();
    let init_bytes_before = bytes_read_counter();
    let (cpu_before, _) = process_resources();
    let started = Instant::now();
    let reader = reopen_fixture(fixture, implementation).await;
    let init_elapsed = started.elapsed();
    let init_iops = iops_counter() - init_iops_before;
    let init_bytes_read = bytes_read_counter() - init_bytes_before;

    let read_iops_before = iops_counter();
    let read_bytes_before = bytes_read_counter();
    let read_started = Instant::now();
    let read_result = read_all_partitions(reader, implementation, concurrency)
        .await
        .expect("read every deterministic partition");
    let read_elapsed = read_started.elapsed();
    let total_elapsed = started.elapsed();
    let (cpu_after, peak_rss_bytes) = process_resources();
    assert_eq!(read_result.rows, fixture.total_rows);

    writeln!(
        std::io::stderr().lock(),
        "two_file_shuffle_read scenario={} implementation={} flush_groups={} partitions={} rows={} cv={:.4} concurrency={} init_ms={:.3} read_ms={:.3} total_ms={:.3} rows_per_second={:.0} cpu_seconds={:.6} peak_rss_mib={:.1} init_scheduler_iops={} init_scheduler_bytes_read={} read_scheduler_iops={} read_scheduler_bytes_read={} logical_partition_reads={} logical_data_ranges={} offset_entries={} per_file_read_calls=unavailable_use_scheduler_trace",
        distribution.name(),
        implementation.name(),
        NUM_FLUSH_GROUPS,
        NUM_PARTITIONS,
        read_result.rows,
        fixture.coefficient_of_variation,
        concurrency,
        init_elapsed.as_secs_f64() * 1_000.0,
        read_elapsed.as_secs_f64() * 1_000.0,
        total_elapsed.as_secs_f64() * 1_000.0,
        read_result.rows as f64 / total_elapsed.as_secs_f64(),
        cpu_after - cpu_before,
        peak_rss_bytes as f64 / (1024.0 * 1024.0),
        init_iops,
        init_bytes_read,
        iops_counter() - read_iops_before,
        bytes_read_counter() - read_bytes_before,
        read_result.windows,
        read_result.windows * NUM_FLUSH_GROUPS,
        NUM_PARTITIONS * NUM_FLUSH_GROUPS,
    )
    .expect("write shuffle benchmark diagnostic");
}

fn bench_two_file_shuffle_read(criterion: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("create benchmark runtime");
    let concurrency = get_num_compute_intensive_cpus();

    let distributions = match std::env::var("LANCE_SHUFFLE_BENCH_DISTRIBUTION").as_deref() {
        Ok("uniform") => vec![Distribution::Uniform],
        Ok("hotspot") => vec![Distribution::Hotspot],
        Ok(value) => panic!("unknown LANCE_SHUFFLE_BENCH_DISTRIBUTION={value}"),
        Err(_) => vec![Distribution::Uniform, Distribution::Hotspot],
    };
    let implementations = match std::env::var("LANCE_SHUFFLE_BENCH_IMPLEMENTATION").as_deref() {
        Ok("baseline") => vec![ReaderImplementation::Baseline],
        Ok("current") => vec![ReaderImplementation::Current],
        Ok(value) => panic!("unknown LANCE_SHUFFLE_BENCH_IMPLEMENTATION={value}"),
        Err(_) => vec![
            ReaderImplementation::Baseline,
            ReaderImplementation::Current,
        ],
    };

    for distribution in distributions {
        let fixture = runtime.block_on(build_fixture(distribution));
        for implementation in implementations.iter().copied() {
            runtime.block_on(print_diagnostic(
                distribution,
                implementation,
                &fixture,
                concurrency,
            ));
            let benchmark_id = format!("{}/{}", distribution.name(), implementation.name());

            let mut reopen_group = criterion.benchmark_group("two_file_shuffle_reopen");
            reopen_group.bench_function(&benchmark_id, |bencher| {
                bencher
                    .to_async(&runtime)
                    .iter(|| async { black_box(reopen_fixture(&fixture, implementation).await) });
            });
            reopen_group.finish();

            let reader = match implementation {
                ReaderImplementation::Baseline => fixture.baseline_reader.clone(),
                ReaderImplementation::Current => fixture.current_reader.clone(),
            };
            let mut read_group = criterion.benchmark_group("two_file_shuffle_read_only");
            read_group.throughput(Throughput::Elements(fixture.total_rows));
            read_group.bench_function(&benchmark_id, |bencher| {
                bencher.to_async(&runtime).iter(|| async {
                    let result = read_all_partitions(reader.clone(), implementation, concurrency)
                        .await
                        .expect("read every deterministic partition");
                    assert_eq!(result.rows, fixture.total_rows);
                    black_box(result.rows)
                });
            });
            read_group.finish();

            let mut total_group = criterion.benchmark_group("two_file_shuffle_reopen_and_read");
            total_group.throughput(Throughput::Elements(fixture.total_rows));
            total_group.bench_function(&benchmark_id, |bencher| {
                bencher.to_async(&runtime).iter(|| async {
                    let reader = reopen_fixture(&fixture, implementation).await;
                    let result = read_all_partitions(reader, implementation, concurrency)
                        .await
                        .expect("read every deterministic partition");
                    assert_eq!(result.rows, fixture.total_rows);
                    black_box(result.rows)
                });
            });
            total_group.finish();
        }
    }
}

fn main() {
    // SAFETY: this is the first action in this single-purpose benchmark binary,
    // before the Tokio runtime or any other worker threads are created.
    unsafe {
        std::env::set_var("LANCE_SHUFFLE_BATCH_BYTES", "1");
    }

    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(8))
        .configure_from_args();
    bench_two_file_shuffle_read(&mut criterion);
    criterion.final_summary();
}
