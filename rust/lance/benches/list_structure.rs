// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Isolated-process list scan measurements. Write fixtures in a separate process,
//! then run the same fixture and arguments on the baseline and changed binaries.
//! Use an S3 URI for I/O comparisons; local files only characterize allocation.
//!
//! ```text
//! cargo build -p lance --bench list_structure --profile release-no-lto
//! <binary> write --uri /tmp/list.lance --kind binary --rows 128 --elements 8 --width 262144
//! <binary> scan --uri s3://<test-bucket>/<owned-prefix>/list.lance --query length
//! ```

#![allow(clippy::print_stdout)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;
#[cfg(unix)]
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use arrow_array::builder::{FixedSizeBinaryBuilder, LargeBinaryBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, Int64Array, ListArray, RecordBatch, RecordBatchIterator};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{Field, Schema};
use clap::{Parser, Subcommand, ValueEnum};
use futures::{StreamExt, TryStreamExt, stream};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance_core::cache::LanceCache;
use lance_encoding::constants::STRUCTURAL_ENCODING_META_KEY;
use lance_encoding::decoder::{DecoderPlugins, PageEncoding};
use lance_encoding::format::metadata::page_layout::Layout as PageLayout;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::LanceFileVersion;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use serde_json::json;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// Count requested Rust allocation bytes separately from resident process memory.
// Zero-filled placeholders may reserve bytes without faulting in physical pages.
struct MeasuredAllocator;
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

fn record_allocation(size: usize) {
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: Every operation forwards the original pointer/layout contract to the
// system allocator. Counters never access the allocation or change its lifetime.
unsafe impl GlobalAlloc for MeasuredAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid allocation layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_allocation(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid allocation layout.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_allocation(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: The pointer and layout are the matching live allocation.
        unsafe { System.dealloc(ptr, layout) };
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The caller satisfies the system allocator's realloc contract.
        let next = unsafe { System.realloc(ptr, layout, new_size) };
        if !next.is_null() {
            if new_size >= layout.size() {
                record_allocation(new_size - layout.size());
            } else {
                LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        next
    }
}

#[global_allocator]
static ALLOCATOR: MeasuredAllocator = MeasuredAllocator;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Inspect {
        #[arg(long)]
        uri: String,
    },
    Write {
        #[arg(long)]
        uri: String,
        #[arg(long, value_enum, default_value = "binary")]
        kind: Kind,
        #[arg(long, default_value = "fullzip")]
        layout: String,
        #[arg(long, default_value_t = 128)]
        rows: u32,
        #[arg(long, default_value_t = 8)]
        elements: u32,
        #[arg(long, default_value_t = 262_144)]
        width: i32,
    },
    Scan {
        #[arg(long)]
        uri: String,
        #[arg(long, value_enum)]
        query: Query,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        #[arg(long, default_value_t = 32)]
        batch_rows: usize,
        #[arg(long, default_value_t = 2)]
        batch_readahead: usize,
        #[arg(long, default_value_t = 134_217_728)]
        io_buffer_bytes: u64,
        #[arg(long, default_value_t = 100)]
        selectivity_percent: u32,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Kind {
    Binary,
    FixedBinary,
    Integer,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Query {
    Length,
    Nonempty,
    Count,
    Mixed,
    Full,
    Scalar,
}

#[cfg(unix)]
fn peak_rss_bytes() -> Result<u64> {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage initializes the provided rusage on a successful call.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: the successful call above initialized every field.
    let usage = unsafe { usage.assume_init() };
    let rss = u64::try_from(usage.ru_maxrss)?;
    Ok(if cfg!(target_os = "macos") {
        rss
    } else {
        rss * 1024
    })
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> Result<u64> {
    Err("Peak RSS measurements require a Unix host".into())
}

fn fixture_batch(
    start: u32,
    rows: u32,
    elements: u32,
    width: i32,
    kind: Kind,
    layout: &str,
) -> Result<RecordBatch> {
    let mut offsets = vec![0_i32];
    let mut validity = Vec::with_capacity(rows as usize);
    for row in start..start + rows {
        let count = if row % 17 == 0 || row % 13 == 0 {
            0
        } else {
            elements
        };
        offsets.push(offsets.last().copied().unwrap_or_default() + i32::try_from(count)?);
        validity.push(row % 17 != 0);
    }
    let count = *offsets.last().ok_or("Missing list offset")? as usize;
    let mut state = u64::from(start) + 0x9e37_79b9;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let values: ArrayRef = match kind {
        Kind::Integer => Arc::new(Int64Array::from(
            (0..count)
                .map(|i| (i % 11 != 0).then(|| next() as i64))
                .collect::<Vec<_>>(),
        )),
        Kind::Binary => {
            let mut builder = LargeBinaryBuilder::with_capacity(count, count * width as usize);
            let mut bytes = vec![0; width as usize];
            for i in 0..count {
                if i % 11 == 0 {
                    builder.append_null();
                } else {
                    for byte in &mut bytes {
                        *byte = next() as u8;
                    }
                    builder.append_value(&bytes);
                }
            }
            Arc::new(builder.finish())
        }
        Kind::FixedBinary => {
            let mut builder = FixedSizeBinaryBuilder::with_capacity(count, width);
            let mut bytes = vec![0; width as usize];
            for i in 0..count {
                if i % 11 == 0 {
                    builder.append_null();
                } else {
                    for byte in &mut bytes {
                        *byte = next() as u8;
                    }
                    builder.append_value(&bytes)?;
                }
            }
            Arc::new(builder.finish())
        }
    };
    let item = Arc::new(Field::new("item", values.data_type().clone(), true));
    let items = Arc::new(ListArray::new(
        item,
        OffsetBuffer::new(offsets.into()),
        values,
        Some(NullBuffer::from(validity)),
    ));
    let ids: ArrayRef = Arc::new(Int64Array::from(
        (start..start + rows).map(i64::from).collect::<Vec<_>>(),
    ));
    let mut other = LargeBinaryBuilder::with_capacity(rows as usize, rows as usize * 16_384);
    let mut bytes = vec![0; 16_384];
    for _ in 0..rows {
        for byte in &mut bytes {
            *byte = next() as u8;
        }
        other.append_value(&bytes);
    }
    let other: ArrayRef = Arc::new(other.finish());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ids.data_type().clone(), false),
        Field::new("items", items.data_type().clone(), true).with_metadata(HashMap::from([(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            layout.to_string(),
        )])),
        Field::new("other", other.data_type().clone(), false),
    ]));
    Ok(RecordBatch::try_new(schema, vec![ids, items, other])?)
}

async fn scan_once(
    dataset: &Dataset,
    query: Query,
    batch_rows: usize,
    batch_readahead: usize,
    io_buffer_bytes: u64,
    selectivity_percent: u32,
) -> Result<serde_json::Value> {
    let mut scanner = dataset.scan();
    scanner
        .batch_size(batch_rows)
        .batch_readahead(batch_readahead)
        .io_buffer_size(io_buffer_bytes);
    match query {
        Query::Length => {
            scanner.project_with_transform(&[("length", "array_length(items)")])?;
        }
        Query::Nonempty => {
            scanner.project(&["id"])?;
        }
        Query::Count => {
            scanner.project::<String>(&[])?.with_row_id();
        }
        Query::Mixed => {
            scanner
                .project_with_transform(&[("other", "other"), ("length", "array_length(items)")])?;
        }
        Query::Full => {
            scanner.project(&["items"])?;
        }
        Query::Scalar => {
            scanner.project(&["id"])?;
        }
    }
    let mut predicates = Vec::new();
    if matches!(query, Query::Nonempty | Query::Count) {
        predicates.push("array_length(items) > 0".to_string());
    }
    if selectivity_percent < 100 {
        predicates.push(format!("id % 100 < {selectivity_percent}"));
    }
    if !predicates.is_empty() {
        scanner.filter(&predicates.join(" AND "))?;
    }
    let start = Instant::now();
    if matches!(query, Query::Count) {
        let rows = scanner.count_rows().await?;
        return Ok(json!({"elapsed_ms": start.elapsed().as_secs_f64()*1000.0,
            "rows":rows, "length_sum":0,"returned_array_bytes":0}));
    }
    let mut batches = scanner.try_into_stream().await?;
    let mut rows = 0_u64;
    let mut length_sum = 0_u64;
    let mut returned_array_bytes = 0_u64;
    while let Some(batch) = batches.try_next().await? {
        rows += batch.num_rows() as u64;
        returned_array_bytes += batch.get_array_memory_size() as u64;
        if let Some(lengths) = batch.column_by_name("length") {
            length_sum += lengths
                .as_primitive::<UInt64Type>()
                .iter()
                .flatten()
                .sum::<u64>();
        }
        black_box(&batch);
    }
    Ok(json!({"elapsed_ms":start.elapsed().as_secs_f64()*1000.0,
        "rows":rows,"length_sum":length_sum,"returned_array_bytes":returned_array_bytes}))
}

#[tokio::main]
async fn main() -> Result<()> {
    match Args::parse().command {
        Command::Inspect { uri } => {
            let dataset = Dataset::open(&uri).await?;
            let store = dataset.object_store(None).await?;
            let scheduler =
                ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
            let mut files = Vec::new();
            for fragment in dataset.get_fragments() {
                for file in &fragment.metadata().files {
                    if file.base_id.is_some() {
                        return Err("fixture inspection requires the primary data location".into());
                    }
                    let path = dataset.data_dir().join(file.path.as_str());
                    let reader = FileReader::try_open(
                        scheduler.open_file(&path, &file.file_size_bytes).await?,
                        None,
                        Arc::<DecoderPlugins>::default(),
                        &LanceCache::no_cache(),
                        FileReaderOptions::default(),
                    )
                    .await?;
                    let mut columns = Vec::new();
                    for (index, column) in reader.metadata().column_infos.iter().enumerate() {
                        let mut layouts = BTreeMap::<&str, usize>::new();
                        let mut rows = Vec::new();
                        for page in column.page_infos.iter() {
                            *layouts.entry(layout_name(&page.encoding)?).or_default() += 1;
                            rows.push(page.num_rows);
                        }
                        columns
                            .push(json!({"column_index":index,"layouts":layouts,"page_rows":rows}));
                    }
                    files.push(json!({"path":file.path,"columns":columns}));
                }
            }
            println!("{}", json!({"uri":uri,"files":files}));
        }
        Command::Write {
            uri,
            kind,
            layout,
            rows,
            elements,
            width,
        } => {
            if rows == 0 || elements == 0 || width <= 0 {
                return Err("rows, elements and width must be positive".into());
            }
            let schema = fixture_batch(0, rows.min(16), elements, width, kind, &layout)?.schema();
            let batch_layout = layout.clone();
            let batches = (0..rows).step_by(16).map(move |start| {
                fixture_batch(
                    start,
                    (rows - start).min(16),
                    elements,
                    width,
                    kind,
                    &batch_layout,
                )
                .map_err(arrow_schema::ArrowError::ExternalError)
            });
            let dataset = Dataset::write(
                RecordBatchIterator::new(batches, schema),
                &uri,
                Some(WriteParams {
                    max_rows_per_file: (rows as usize).div_ceil(4),
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await?;
            println!(
                "{}",
                json!({"operation":"write","uri":uri,"kind":format!("{kind:?}"),
                "layout":layout,"rows":rows,"elements":elements,"width":width,
                "fragments":dataset.get_fragments().len(),"version":dataset.version().version})
            );
        }
        Command::Scan {
            uri,
            query,
            iterations,
            concurrency,
            batch_rows,
            batch_readahead,
            io_buffer_bytes,
            selectivity_percent,
        } => {
            if iterations == 0
                || concurrency == 0
                || batch_rows == 0
                || batch_readahead == 0
                || !(1..=100).contains(&selectivity_percent)
            {
                return Err(
                    "iterations, concurrency, batch_rows and batch_readahead must be positive; selectivity_percent must be in 1..=100"
                        .into(),
                );
            }
            let dataset = Arc::new(Dataset::open(&uri).await?);
            let store = dataset.object_store(None).await?;
            let peak_after_open = peak_rss_bytes()?;
            for iteration in 0..iterations {
                store.io_stats_incremental();
                let live_before = LIVE_BYTES.load(Ordering::Relaxed);
                PEAK_BYTES.store(live_before, Ordering::Relaxed);
                let start = Instant::now();
                let results = stream::iter(0..concurrency)
                    .map(|_| {
                        scan_once(
                            &dataset,
                            query,
                            batch_rows,
                            batch_readahead,
                            io_buffer_bytes,
                            selectivity_percent,
                        )
                    })
                    .buffer_unordered(concurrency)
                    .try_collect::<Vec<_>>()
                    .await?;
                let elapsed = start.elapsed().as_secs_f64();
                let stats = store.io_stats_incremental();
                println!(
                    "{}",
                    json!({"operation":"scan","uri":uri,"query":format!("{query:?}"),
                    "iteration":iteration,"session":if iteration==0 {"new"} else {"reused"},
                    "concurrency":concurrency,"batch_rows":batch_rows,"batch_readahead":batch_readahead,
                    "io_buffer_bytes":io_buffer_bytes,"selectivity_percent":selectivity_percent,"elapsed_ms":elapsed*1000.0,
                    "queries_per_second":concurrency as f64/elapsed,"read_bytes":stats.read_bytes,
                    "read_iops":stats.read_iops,"peak_rss_bytes":peak_rss_bytes()?,
                    "io_stages":stats.num_stages,
                    "rust_live_bytes_before":live_before,"rust_live_bytes_after":LIVE_BYTES.load(Ordering::Relaxed),
                    "rust_peak_live_bytes":PEAK_BYTES.load(Ordering::Relaxed),
                    "peak_rss_after_open_bytes":peak_after_open,"queries":results})
                );
            }
        }
    }
    Ok(())
}

fn layout_name(encoding: &PageEncoding) -> Result<&'static str> {
    let layout = match encoding {
        PageEncoding::Legacy(_) => return Ok("array"),
        PageEncoding::Structural(layout) => Cow::Borrowed(layout),
        PageEncoding::Planar(layout) if layout.sparse().is_some() => return Ok("sparse"),
        PageEncoding::Planar(layout) => Cow::Owned(layout.dense_execution_layout()?),
    };
    Ok(match layout.layout.as_ref().ok_or("Missing page layout")? {
        PageLayout::FullZipLayout(_) => "fullzip",
        PageLayout::MiniBlockLayout(_) => "miniblock",
        PageLayout::ConstantLayout(_) => "constant",
        PageLayout::BlobLayout(_) => "blob",
    })
}
