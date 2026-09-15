// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// This bench's output *is* its result, so it prints rather than logs.
#![allow(clippy::print_stdout)]
//! Reports the resident cost of scheduling sparse structural pages.
//!
//! This is a measurement harness, not a timing benchmark, so it does not use Criterion:
//! the quantities of interest are exact rather than statistical.
//!
//! Two numbers are reported per input:
//!
//! * `cache_bytes` — what [`LanceCache::size_bytes`] holds after the scheduler has
//!   initialized. This is the state that stays resident for the lifetime of an open
//!   dataset, once per page per column, so it is the number that decides how much memory
//!   a wide 2.3 table costs to keep open.
//! * `init_bytes` — total bytes passed to the global allocator while initializing. This
//!   catches transient allocations that never reach the cache, which `cache_bytes` cannot
//!   see by construction.
//!
//! Run:
//!
//! ```text
//! cargo bench -p lance-encoding --bench sparse_footprint
//! cargo bench -p lance-encoding --bench sparse_footprint -- --json
//! ```
//!
//! `--json` emits one object per line for `ci/sparse_ab.py` to diff across two builds.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use lance_core::cache::LanceCache;
use lance_encoding::{
    decoder::{DecodeBatchScheduler, DecoderConfig, DecoderPlugins, FilterExpression},
    encoder::EncodedBatch,
};

#[path = "sparse/cases.rs"]
mod cases;

use cases::{Case, cases, encode, layouts, visible_items};

/// Counting allocator. Recording is opt-in so that only the region under test is counted;
/// otherwise input construction and encoding would dominate the totals.
struct Counting;

static RECORDING: AtomicBool = AtomicBool::new(false);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if RECORDING.load(Ordering::Relaxed) {
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if RECORDING.load(Ordering::Relaxed) && new_size > layout.size() {
            ALLOC_BYTES.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `f` with allocation recording on, returning `(bytes, count)`.
///
/// Single-threaded by construction: the scheduler runs on a current-thread runtime so no
/// other thread's allocations can leak into the totals.
fn measure_allocations<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    RECORDING.store(true, Ordering::Relaxed);
    let out = f();
    RECORDING.store(false, Ordering::Relaxed);
    (
        out,
        ALLOC_BYTES.load(Ordering::Relaxed),
        ALLOC_COUNT.load(Ordering::Relaxed),
    )
}

/// Initialize a scheduler against a cold cache and report what the cache retains.
///
/// `DecodeBatchScheduler::try_new` is what calls `initialize` on the field schedulers, so
/// this measures exactly the work an open dataset does once per page.
fn cache_footprint(encoded: &EncodedBatch) -> Footprint {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");
    // Capacity well above anything the matrix produces, so nothing is evicted and
    // `size_bytes` reflects the full retained state rather than a capacity ceiling.
    let cache = Arc::new(LanceCache::with_capacity(4 * 1024 * 1024 * 1024));
    let io = Arc::new(lance_encoding::BufferScheduler::new(encoded.data.clone()))
        as Arc<dyn lance_encoding::EncodingsIo>;
    let filter = FilterExpression::no_filter();

    let (scheduler, alloc_bytes, alloc_count) = measure_allocations(|| {
        rt.block_on(DecodeBatchScheduler::try_new(
            encoded.schema.as_ref(),
            &encoded.top_level_columns,
            &encoded.page_table,
            &vec![],
            encoded.num_rows,
            Arc::<DecoderPlugins>::default(),
            io,
            cache.clone(),
            &filter,
            &DecoderConfig::default(),
        ))
        .expect("scheduler")
    });
    drop(scheduler);

    Footprint {
        cache_bytes: rt.block_on(cache.size_bytes()) as u64,
        init_bytes: alloc_bytes,
        init_allocs: alloc_count,
    }
}

struct Footprint {
    cache_bytes: u64,
    init_bytes: u64,
    init_allocs: u64,
}

struct Row {
    case: &'static str,
    note: &'static str,
    layout: String,
    pages: usize,
    rows: u64,
    items: u64,
    /// `None` for an empty page: `DecodeBatchScheduler::try_new` requires at least one row,
    /// so there is no scheduler state to measure.
    footprint: Option<Footprint>,
}

fn run(case: &Case) -> Row {
    let encoded = encode(case);
    let observed = layouts(&encoded);
    let layout = observed
        .first()
        .map(|l| l.to_string())
        .unwrap_or_else(|| "none".to_string());
    Row {
        case: case.name,
        note: case.note,
        layout,
        pages: observed.len(),
        rows: encoded.num_rows,
        items: visible_items(&encoded),
        footprint: (encoded.num_rows > 0).then(|| cache_footprint(&encoded)),
    }
}

fn main() {
    let json = std::env::args().any(|a| a == "--json");

    let rows: Vec<Row> = cases().iter().map(run).collect();

    if json {
        for row in &rows {
            let Some(footprint) = &row.footprint else {
                continue;
            };
            println!(
                r#"{{"case":"{}","layout":"{}","pages":{},"rows":{},"items":{},"cache_bytes":{},"init_bytes":{},"init_allocs":{}}}"#,
                row.case,
                row.layout,
                row.pages,
                row.rows,
                row.items,
                footprint.cache_bytes,
                footprint.init_bytes,
                footprint.init_allocs,
            );
        }
        return;
    }

    println!("sparse structural scheduler: resident cost of an initialized scheduler\n");
    println!(
        "{:<30} {:>9} {:>6} {:>10} {:>12} {:>12} {:>11} {:>10}",
        "case",
        "layout",
        "pages",
        "items",
        "cache_bytes",
        "bytes/page",
        "init_bytes",
        "init_allocs"
    );
    println!("{}", "-".repeat(107));
    for row in &rows {
        let Some(footprint) = &row.footprint else {
            println!(
                "{:<30} {:>9} {:>6} {:>10} {:>12} {:>12} {:>11} {:>10}",
                row.case, row.layout, row.pages, row.items, "-", "-", "-", "-"
            );
            continue;
        };
        let per_page = footprint.cache_bytes / row.pages.max(1) as u64;
        println!(
            "{:<30} {:>9} {:>6} {:>10} {:>12} {:>12} {:>11} {:>10}",
            row.case,
            row.layout,
            row.pages,
            row.items,
            footprint.cache_bytes,
            per_page,
            footprint.init_bytes,
            footprint.init_allocs,
        );
    }

    println!("\nnotes");
    for row in &rows {
        println!("  {:<30} {} ({} rows)", row.case, row.note, row.rows);
    }
}
