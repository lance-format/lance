// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zone map builds cut zones at the configured physical boundaries, reuse
//! the write seeds fragments already carry where whole records fit a zone,
//! and scan only the rows the seeds do not cover. Every claim about what was reused or scanned is
//! taken from the build itself, and every query is compared with a plain
//! scan by row identity.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_array::{
    Array, Float64Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::utils::tempfile::TempStrDir;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::pbold;
use lance_index::scalar::seed::SEED_META_KEY_PREFIX;
use lance_index::scalar::zonemap::{ZoneMapIndex, ZoneSummary};
use lance_index::scalar::{BuiltinIndexType, SargableQuery, ScalarIndexParams};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_io::utils::tracking_store::IoStats;
use lance_table::format::{DataFile, IndexMetadata};
use roaring::RoaringBitmap;

use crate::Dataset;
use crate::dataset::WriteParams;
use crate::dataset::optimize::{CompactionMode, CompactionOptions, compact_files};
use crate::index::scalar::zonemap::{SEED_REUSE_REPORTS, SeedReuseReport, harvest_concurrency};
use crate::index::{CreateIndexBuilder, DatasetIndexExt, DatasetIndexInternalExt};

const ROWS_PER_FILE: usize = 20000;
/// Rows per written fragment.
const N: i32 = ROWS_PER_FILE as i32;

fn index_name(column: &str) -> String {
    format!("{column}_zones")
}

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("val", DataType::Int64, true),
        ArrowField::new("name", DataType::Utf8, true),
        ArrowField::new("x", DataType::Float64, true),
    ]))
}

/// `val` is null every fifth row and otherwise a scrambled multiple of the
/// id, so zones overlap and most predicates prune some but not all of them.
fn val(i: i32) -> Option<i64> {
    (i % 5 != 0).then(|| i64::from(i) * 7 % 1013 + i64::from(i) / 100 * 1000)
}

fn name(i: i32) -> Option<String> {
    (i % 11 != 0).then(|| format!("n{:05}", (i * 37) % 20000))
}

fn x(i: i32) -> Option<f64> {
    match i % 13 {
        0 => None,
        1 => Some(f64::NAN),
        2 => Some(-f64::NAN),
        3 => Some(f64::INFINITY),
        4 => Some(f64::NEG_INFINITY),
        _ => Some(f64::from(i % 97) - 48.0),
    }
}

fn rows(ids: std::ops::Range<i32>) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(ids.clone())),
            Arc::new(Int64Array::from_iter(ids.clone().map(val))),
            Arc::new(StringArray::from_iter(ids.clone().map(name))),
            Arc::new(Float64Array::from_iter(ids.map(x))),
        ],
    )
    .unwrap()
}

fn zone_map_params(
    rows_per_zone: u64,
    use_seeds: bool,
    store_null_bitmap: bool,
) -> ScalarIndexParams {
    ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(&serde_json::json!({
        "rows_per_zone": rows_per_zone,
        "use_seeds": use_seeds,
        "store_null_bitmap": store_null_bitmap,
    }))
}

async fn write(uri: &str, ids: std::ops::Range<i32>) -> Dataset {
    Dataset::write(
        RecordBatchIterator::new([Ok(rows(ids))], schema()),
        uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FILE,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

async fn append(dataset: &mut Dataset, ids: std::ops::Range<i32>) {
    dataset
        .append(RecordBatchIterator::new([Ok(rows(ids))], schema()), None)
        .await
        .unwrap();
}

/// Create (or replace) the zone map on `column`, over `fragments` when given.
async fn create_index(
    dataset: &mut Dataset,
    column: &str,
    params: &ScalarIndexParams,
    fragments: Option<Vec<u32>>,
) {
    let mut builder = CreateIndexBuilder::new(dataset, &[column], IndexType::ZoneMap, params)
        .name(index_name(column))
        .replace(true);
    if let Some(fragments) = fragments {
        builder = builder.fragments(fragments);
    }
    builder.await.unwrap();
}

/// Run `build` and return the seed reuse reports it produced.
async fn with_reports<F, T>(build: F) -> (T, Vec<SeedReuseReport>)
where
    F: std::future::Future<Output = T>,
{
    SEED_REUSE_REPORTS
        .scope(Mutex::new(Vec::new()), async {
            let out = build.await;
            let reports = SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone());
            (out, reports)
        })
        .await
}

async fn index_metadata(dataset: &Dataset, column: &str) -> Vec<IndexMetadata> {
    dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|index| index.name == index_name(column))
        .cloned()
        .collect()
}

fn details(segment: &IndexMetadata) -> pbold::ZoneMapIndexDetails {
    segment
        .index_details
        .as_ref()
        .unwrap()
        .to_msg::<pbold::ZoneMapIndexDetails>()
        .unwrap()
}

fn coverage(segment: &IndexMetadata) -> RoaringBitmap {
    segment.fragment_bitmap.clone().unwrap()
}

/// Ids matching `filter`, through the index or through a plain scan. With
/// `expect_index` the plan must show the zone map being consulted.
async fn ids_matching(
    dataset: &Dataset,
    filter: &str,
    use_index: bool,
    expect_index: bool,
) -> BTreeSet<i32> {
    let mut scanner = dataset.scan();
    scanner
        .filter(filter)
        .unwrap()
        .project(&["id"])
        .unwrap()
        .use_scalar_index(use_index);
    if use_index && expect_index {
        let plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery"),
            "{filter}: the zone map was not used:\n{plan}"
        );
    }
    let batch = scanner.try_into_batch().await.unwrap();
    batch["id"]
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .values()
        .iter()
        .copied()
        .collect()
}

const VAL_FILTERS: &[&str] = &[
    "val > 500",
    "val < 2300",
    "val >= 7000 AND val <= 7300",
    "val = 707",
    "val IN (14, 20000, 8191)",
    "val IS NULL",
    "val IS NOT NULL",
    "val BETWEEN 100 AND 110",
];

/// Every filter yields the same ids with and without the index. The index
/// must appear in the plan, except for `IS NOT NULL` on an index without a
/// null bitmap, which the planner leaves to the ordinary filter.
async fn assert_index_matches_scan(dataset: &Dataset, filters: &[&str]) {
    assert_index_matches_scan_with(dataset, filters, true).await;
}

async fn assert_index_matches_scan_with(
    dataset: &Dataset,
    filters: &[&str],
    has_null_bitmap: bool,
) {
    for filter in filters {
        let expect_index = has_null_bitmap || !filter.contains("IS NOT NULL");
        let indexed = ids_matching(dataset, filter, true, expect_index).await;
        let scanned = ids_matching(dataset, filter, false, false).await;
        assert_eq!(indexed, scanned, "{filter}");
    }
}

fn io(dataset: &Dataset) -> IoStats {
    dataset.object_store.io_stats_incremental()
}

async fn open_data_file(dataset: &Dataset, file: &DataFile) -> FileReader {
    let path = dataset
        .data_file_dir(file)
        .unwrap()
        .join(file.path.as_str());
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await
        .unwrap();
    FileReader::try_open(
        file_scheduler,
        None,
        Default::default(),
        &dataset.metadata_cache.file_metadata_cache(&path),
        FileReaderOptions::default(),
    )
    .await
    .unwrap()
}

async fn seed_locator(dataset: &Dataset, fragment_id: u64, column: &str) -> Option<String> {
    let fragment = dataset
        .fragments()
        .iter()
        .find(|f| f.id == fragment_id)
        .unwrap()
        .clone();
    let reader = open_data_file(dataset, fragment.files.last().unwrap()).await;
    reader
        .metadata()
        .file_schema
        .metadata
        .get(&format!("{SEED_META_KEY_PREFIX}{column}"))
        .cloned()
}

/// Bytes read from the dataset's data files, split into the column data
/// page section and everything else (footer, column metadata and global
/// buffers such as seeds). Reads of other objects are ignored.
/// Per data file: (column page bytes, footer/metadata/seed bytes) read.
async fn classify_data_file_reads(dataset: &Dataset, io: &IoStats) -> HashMap<String, (u64, u64)> {
    // Pages and global buffers (seeds) share the region before the column
    // metadata; the global buffers are carved out of it.
    let mut layout = HashMap::new();
    for fragment in dataset.fragments().iter() {
        for file in &fragment.files {
            let path = dataset
                .data_file_dir(file)
                .unwrap()
                .join(file.path.as_str());
            let metadata = open_data_file(dataset, file).await.metadata().clone();
            let data_end = metadata.num_data_bytes + metadata.num_global_buffer_bytes;
            let buffers: Vec<std::ops::Range<u64>> = metadata
                .file_buffers
                .iter()
                .map(|buffer| buffer.position..buffer.position + buffer.size)
                .collect();
            layout.insert(path.to_string(), (data_end, buffers));
        }
    }
    fn overlap(a: &std::ops::Range<u64>, b: &std::ops::Range<u64>) -> u64 {
        a.end.min(b.end).saturating_sub(a.start.max(b.start))
    }
    let mut per_file: HashMap<String, (u64, u64)> = HashMap::new();
    for request in &io.requests {
        let Some(range) = &request.range else {
            continue;
        };
        let Some((data_end, buffers)) = layout.get(&request.path.to_string()) else {
            continue;
        };
        let in_region = overlap(range, &(0..*data_end));
        let in_buffers: u64 = buffers.iter().map(|buffer| overlap(range, buffer)).sum();
        let in_pages = in_region - in_buffers;
        let entry = per_file.entry(request.path.to_string()).or_default();
        entry.0 += in_pages;
        entry.1 += (range.end - range.start) - in_pages;
    }
    per_file
}

/// Column page bytes read from each fragment's serving file for `column`.
fn pages_per_fragment(
    dataset: &Dataset,
    column: &str,
    per_file: &HashMap<String, (u64, u64)>,
) -> Vec<(u64, u64)> {
    let field = dataset.schema().field(column).unwrap().id;
    dataset
        .fragments()
        .iter()
        .map(|fragment| {
            let file = fragment
                .files
                .iter()
                .rev()
                .find(|file| file.fields.contains(&field))
                .unwrap();
            let path = dataset
                .data_file_dir(file)
                .unwrap()
                .join(file.path.as_str())
                .to_string();
            (fragment.id, per_file.get(&path).map_or(0, |v| v.0))
        })
        .collect()
}

/// Seeds written with `record_rows` by an index on `column` for every
/// fragment but the first: create a small seeded index on the first
/// fragment, then append `appends` fragments.
async fn seeded_dataset(uri: &str, record_rows: u64, appends: i32, bitmap: bool) -> Dataset {
    let mut dataset = write(uri, 0..N).await;
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(record_rows, true, bitmap),
        None,
    )
    .await;
    for k in 1..=appends {
        append(&mut dataset, k * N..(k + 1) * N).await;
    }
    dataset
}

/// The same data written without any index.
async fn unseeded_dataset(uri: &str, appends: i32) -> Dataset {
    let mut dataset = write(uri, 0..N).await;
    for k in 1..=appends {
        append(&mut dataset, k * N..(k + 1) * N).await;
    }
    dataset
}

async fn zone_summaries(dataset: &Dataset, column: &str) -> Vec<ZoneSummary> {
    let segments = index_metadata(dataset, column).await;
    let mut out = Vec::new();
    for segment in &segments {
        let index = dataset
            .open_scalar_index(column, &segment.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        out.extend(
            index
                .as_any()
                .downcast_ref::<ZoneMapIndex>()
                .unwrap()
                .zone_summaries(),
        );
    }
    out.sort_by_key(|z| (z.bound.fragment_id, z.bound.start));
    out
}

fn layout(zones: &[ZoneSummary]) -> Vec<(u64, u64, usize)> {
    zones
        .iter()
        .map(|z| (z.bound.fragment_id, z.bound.start, z.bound.length))
        .collect()
}

/// The configured layout of a fragment of `rows` physical rows.
fn target_layout(fragment: u64, rows: u64, t: u64) -> Vec<(u64, u64, usize)> {
    (0..rows.div_ceil(t))
        .map(|k| {
            (
                fragment,
                k * t,
                ((k + 1) * t).min(rows) as usize - (k * t) as usize,
            )
        })
        .collect()
}

/// Rows of a fragment of `rows` rows with `s`-row seed records that must be
/// scanned for target zones of `t`: the records crossing a boundary.
fn crossing_rows(rows: u64, s: u64, t: u64) -> u64 {
    let mut scanned = 0;
    let mut start = 0;
    while start < rows {
        let end = (start + s).min(rows);
        if start / t != (end - 1) / t {
            scanned += end - start;
        }
        start = end;
    }
    scanned
}

/// Seed-assisted builds cut exactly the configured zones and produce the
/// same statistics as a plain scan of the same data, for divisible,
/// non-divisible, whole-fragment and phase-shifting target sizes, with and
/// without a null bitmap. Reads are attributed per fragment file.
#[rstest::rstest]
#[case::divisible(64, 320, true)]
#[case::whole_fragment(64, 20000, true)]
#[case::whole_fragment_no_bitmap(64, 20000, false)]
#[case::non_divisible_capped(64, 100, true)]
#[case::non_divisible_partial(64, 160, false)]
#[case::phase_shifting(64, 10000, false)]
#[case::same_size(64, 64, true)]
#[tokio::test]
async fn test_create_index_keeps_configured_boundaries(
    #[case] s: u64,
    #[case] t: u64,
    #[case] bitmap: bool,
) {
    let seeded_dir = TempStrDir::default();
    let mut seeded = seeded_dataset(seeded_dir.as_str(), s, 5, bitmap).await;
    let control_dir = TempStrDir::default();
    let mut control = unseeded_dataset(control_dir.as_str(), 5).await;

    io(&seeded);
    let (_, reports) = with_reports(create_index(
        &mut seeded,
        "val",
        &zone_map_params(t, true, bitmap),
        None,
    ))
    .await;
    let seeded_reads = classify_data_file_reads(&seeded, &io(&seeded)).await;
    io(&control);
    create_index(&mut control, "val", &zone_map_params(t, true, bitmap), None).await;
    let control_reads = classify_data_file_reads(&control, &io(&control)).await;

    // Layout and statistics identical to the scan, zones exactly the target.
    let from_seeds = zone_summaries(&seeded, "val").await;
    let from_scan = zone_summaries(&control, "val").await;
    assert_eq!(from_seeds, from_scan, "{s} -> {t}");
    let expected: Vec<(u64, u64, usize)> = (0..6u64)
        .flat_map(|f| target_layout(f, N as u64, t))
        .collect();
    assert_eq!(layout(&from_seeds), expected);

    // The report matches the interval arithmetic: fragment 0 (no seed) is
    // scanned whole, the others reuse whole records and scan crossings.
    // `val` is a narrow column: past half its rows in crossing records a
    // fragment is scanned whole rather than in scattered intervals.
    let report = &reports[0];
    let crossing = crossing_rows(N as u64, s, t);
    let rescanned = if crossing * 2 > N as u64 {
        N as u64
    } else {
        crossing
    };
    assert_eq!(report.scanned_rows, N as u64 + 5 * rescanned, "{s} -> {t}");
    if rescanned == 0 {
        assert_eq!(report.reused_fragments, vec![1, 2, 3, 4, 5]);
        assert_eq!(report.scanned_fragments, vec![0]);
    } else if rescanned == N as u64 {
        assert!(report.reused_fragments.is_empty());
        assert!(report.partial_fragments.is_empty());
        assert_eq!(report.scanned_fragments, vec![0, 1, 2, 3, 4, 5]);
    } else {
        assert_eq!(report.partial_fragments, vec![1, 2, 3, 4, 5]);
        assert_eq!(report.scanned_fragments, vec![0, 1, 2, 3, 4, 5]);
    }
    assert!(report.masked);
    assert_eq!(report.scans, 1, "six fragments fit one scan batch");
    // Seeds are read through a bounded window and dropped once folded: the
    // raw bytes held at once never exceed that many seeds, and the window
    // is never wider than the configured harvest concurrency.
    let seeds_in_window = report.harvest_concurrency.min(5);
    assert!(
        report.max_live_seed_bytes <= seeds_in_window * report.total_seed_bytes.div_ceil(5),
        "{} live seed bytes, {} total over 5 seeds, window {seeds_in_window}",
        report.max_live_seed_bytes,
        report.total_seed_bytes
    );
    assert!(report.max_concurrent_harvests <= report.harvest_concurrency);

    // Column pages: the unseeded fragment reads its pages; a fully reused
    // fragment reads none (below the file-open tail prefetch); a partially
    // reused one reads fewer than the all-scan control read for the same file.
    let seeded_pages = pages_per_fragment(&seeded, "val", &seeded_reads);
    let control_pages = pages_per_fragment(&control, "val", &control_reads);
    for ((fragment, pages), (_, full)) in seeded_pages.iter().zip(&control_pages) {
        if *fragment == 0 {
            assert!(*pages > 4096, "fragment 0 scanned: {pages}");
        } else if rescanned == 0 {
            assert!(
                *pages < 4096,
                "fragment {fragment} reused but read {pages} page bytes"
            );
        } else if rescanned < N as u64 {
            assert!(
                *pages < *full,
                "fragment {fragment}: partial {pages} vs full {full}"
            );
        }
    }
    assert_index_matches_scan_with(&seeded, VAL_FILTERS, bitmap).await;
    if bitmap {
        let segments = index_metadata(&seeded, "val").await;
        let index = seeded
            .open_scalar_index("val", &segments[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let result = index
            .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
            .await
            .unwrap();
        assert!(result.is_exact());
    }
}

/// A fragment smaller than the target is one zone from its records; a seed
/// record larger than the target is never reused.
#[tokio::test]
async fn test_fragment_smaller_than_target_and_seed_larger_than_target() {
    let dir = TempStrDir::default();
    let mut dataset = seeded_dataset(dir.as_str(), 64, 2, true).await;
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(50_000, true, true),
        None,
    ))
    .await;
    assert_eq!(reports[0].reused_fragments, vec![1, 2]);
    assert_eq!(reports[0].scanned_fragments, vec![0]);
    assert_eq!(
        layout(&zone_summaries(&dataset, "val").await),
        vec![(0, 0, N as usize), (1, 0, N as usize), (2, 0, N as usize)]
    );
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;

    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(32, true, true),
        None,
    ))
    .await;
    // Seed records larger than the target: no full record fits a zone. The
    // short tail record [19968, 20000) of each seeded fragment is the one
    // exception (32 rows, exactly zone 624); keeping it would leave all
    // other rows of a narrow column to a scattered scan, so the fragments
    // are scanned whole.
    assert!(reports[0].reused_fragments.is_empty());
    assert!(reports[0].partial_fragments.is_empty());
    assert_eq!(reports[0].scanned_fragments, vec![0, 1, 2]);
    assert_eq!(reports[0].scanned_rows, 3 * N as u64);
    assert_eq!(reports[0].reused_records, 0);
    assert_eq!(
        layout(&zone_summaries(&dataset, "val").await),
        (0..3u64)
            .flat_map(|f| target_layout(f, N as u64, 32))
            .collect::<Vec<_>>()
    );
}

/// Deletions: a deleted fragment is reused only when every zone comes from
/// whole records (physical population), scanned whole when a crossing
/// record would need a partial scan, and a plain scan keeps the configured
/// boundaries past a deleted tail.
#[tokio::test]
async fn test_deletions_keep_boundaries_and_populations_apart() {
    let dir = TempStrDir::default();
    let mut dataset = seeded_dataset(dir.as_str(), 64, 2, true).await;
    // Fragment 1: a hole inside a record; fragment 2: the whole physical tail
    // (last 1,000 rows) and every third row.
    dataset
        .delete(&format!("id >= {} AND id < {}", N + 100, N + 150))
        .await
        .unwrap();
    dataset
        .delete(&format!(
            "id >= {} OR (id >= {} AND id % 3 = 0)",
            3 * N - 1000,
            2 * N
        ))
        .await
        .unwrap();
    assert!(dataset.fragments()[1].deletion_file.is_some());
    assert!(dataset.fragments()[2].deletion_file.is_some());

    // Divisible target: both deleted fragments fully from seeds (physical).
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(640, true, true),
        None,
    ))
    .await;
    assert_eq!(reports[0].reused_fragments, vec![1, 2]);
    assert_eq!(reports[0].scanned_fragments, vec![0]);
    let zones = zone_summaries(&dataset, "val").await;
    assert_eq!(
        layout(&zones),
        (0..3u64)
            .flat_map(|f| target_layout(f, N as u64, 640))
            .collect::<Vec<_>>()
    );
    // Physical counts: fragment 2's tail zone still has its seed's nulls.
    let tail = zones
        .iter()
        .find(|z| z.bound.fragment_id == 2 && z.bound.start == 19_200)
        .unwrap();
    assert!(tail.null_count > 0);
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;

    // Non-divisible target: a crossing record would mix populations, so the
    // deleted fragments are scanned whole (live population) and keep the
    // configured boundaries; the fully deleted tail zone of fragment 2 is
    // not emitted by the scan.
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    ))
    .await;
    assert!(reports[0].reused_fragments.is_empty());
    assert!(reports[0].partial_fragments.is_empty());
    assert_eq!(reports[0].scanned_fragments, vec![0, 1, 2]);
    let zones = zone_summaries(&dataset, "val").await;
    let mut expected: Vec<(u64, u64, usize)> = (0..3u64)
        .flat_map(|f| target_layout(f, N as u64, 1000))
        .collect();
    expected.retain(|(f, start, _)| !(*f == 2 && *start == 19_000));
    assert_eq!(layout(&zones), expected);
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;

    // The same on an unseeded copy: identical layout from the scan.
    let control_dir = TempStrDir::default();
    let mut control = unseeded_dataset(control_dir.as_str(), 2).await;
    control
        .delete(&format!("id >= {} AND id < {}", N + 100, N + 150))
        .await
        .unwrap();
    control
        .delete(&format!(
            "id >= {} OR (id >= {} AND id % 3 = 0)",
            3 * N - 1000,
            2 * N
        ))
        .await
        .unwrap();
    create_index(
        &mut control,
        "val",
        &zone_map_params(1000, true, true),
        None,
    )
    .await;
    assert_eq!(zone_summaries(&control, "val").await, zones);
}

/// One assembly with a partial fragment, a whole-scan fragment (no seed)
/// and a fully reused one: the mask holds the whole fragment too.
#[tokio::test]
async fn test_mixed_assembly_masks_partial_intervals_and_whole_fragments() {
    let dir = TempStrDir::default();
    let mut dataset = seeded_dataset(dir.as_str(), 64, 1, true).await;
    dataset.drop_index(&index_name("val")).await.unwrap();
    append(&mut dataset, 2 * N..3 * N).await; // no seed
    create_index(&mut dataset, "val", &zone_map_params(640, true, true), None).await; // seeds 640
    append(&mut dataset, 3 * N..4 * N).await; // 640-row seed
    dataset.drop_index(&index_name("val")).await.unwrap();
    // Target 1000: fragment 1 (64-row records) is partial; fragments 0 and
    // 2 have no seed; fragment 3 (640-row records) would rescan 60 percent
    // of a narrow column in scattered intervals and is scanned whole.
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    ))
    .await;
    assert!(crossing_rows(N as u64, 640, 1000) * 2 > N as u64);
    assert_eq!(reports[0].partial_fragments, vec![1]);
    assert_eq!(reports[0].scanned_fragments, vec![0, 1, 2, 3]);
    assert!(reports[0].masked);
    assert_eq!(
        reports[0].scanned_rows,
        3 * N as u64 + crossing_rows(N as u64, 64, 1000)
    );
    let control_dir = TempStrDir::default();
    let mut control = unseeded_dataset(control_dir.as_str(), 3).await;
    create_index(
        &mut control,
        "val",
        &zone_map_params(1000, true, true),
        None,
    )
    .await;
    assert_eq!(
        zone_summaries(&dataset, "val").await,
        zone_summaries(&control, "val").await
    );
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
}

/// The harvester resolves the field path it is given: a top-level name
/// that needs quoting finds its column, quoted or not, and a nested path
/// has no seed of its own.
#[tokio::test]
async fn test_harvester_resolves_field_paths() {
    use crate::index::append::SeedHarvester;
    use arrow_array::{ArrayRef, StructArray};
    let child = Arc::new(ArrowField::new("b", DataType::Int64, false));
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("user-id", DataType::Int64, false),
        ArrowField::new(
            "a",
            DataType::Struct(vec![child.as_ref().clone()].into()),
            false,
        ),
    ]));
    let batch = |start: i64| {
        let values = || Arc::new(Int64Array::from_iter_values(start..start + 1000)) as ArrayRef;
        RecordBatch::try_new(
            schema.clone(),
            vec![
                values(),
                Arc::new(StructArray::from(vec![(child.clone(), values())])),
            ],
        )
        .unwrap()
    };
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch(0))], schema.clone()),
        dir.as_str(),
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["user-id"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(100, true, false),
            false,
        )
        .await
        .unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(batch(1000))], schema.clone()),
            None,
        )
        .await
        .unwrap();

    assert!(SeedHarvester::for_column(&dataset, "`user-id`").is_some());
    assert!(SeedHarvester::for_column(&dataset, "user-id").is_some());
    assert!(SeedHarvester::for_column(&dataset, "a.b").is_none());
    assert!(SeedHarvester::for_column(&dataset, "missing").is_none());

    // The create path formats the column as a quoted path and still reuses
    // the appended fragment's seed.
    let (_, reports) = with_reports(async {
        dataset
            .create_index(
                &["user-id"],
                IndexType::ZoneMap,
                None,
                &zone_map_params(100, true, false),
                true,
            )
            .await
            .unwrap();
    })
    .await;
    assert_eq!(reports[0].reused_fragments, vec![1]);
    assert_eq!(reports[0].scanned_fragments, vec![0]);
}

/// A zero zone size is refused before any segment is written, also for a
/// build that scans nothing.
#[tokio::test]
async fn test_zero_rows_per_zone_is_rejected_without_a_scan() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    let err = CreateIndexBuilder::new(
        &mut dataset,
        &["val"],
        IndexType::ZoneMap,
        &zone_map_params(0, true, false),
    )
    .name(index_name("val"))
    .fragments(vec![])
    .await
    .unwrap_err();
    assert!(err.to_string().contains("rows_per_zone"), "{err}");
    assert!(index_metadata(&dataset, "val").await.is_empty());
}

/// Fragments with rows left to read are scanned in batches as they are
/// decided, not one scan per fragment and not one scan after the last
/// decision: 33 seedless fragments take three scans of at most 16.
#[tokio::test]
async fn test_scans_are_batched_by_decided_fragments() {
    let dir = TempStrDir::default();
    let mut dataset = unseeded_dataset(dir.as_str(), 32).await;
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, false),
        None,
    ))
    .await;
    let report = &reports[0];
    assert_eq!(report.scanned_fragments.len(), 33);
    assert_eq!(report.scans, 3);
    assert_eq!(
        layout(&zone_summaries(&dataset, "val").await),
        (0..33u64)
            .flat_map(|f| target_layout(f, N as u64, 1000))
            .collect::<Vec<_>>()
    );
    assert_index_matches_scan_with(&dataset, VAL_FILTERS, false).await;
}

/// Legacy (V1) storage cannot take a physical mask: seeded fragments are
/// scanned whole without a mask and statistics equal the scan.
#[tokio::test]
async fn test_legacy_storage_scans_without_a_mask() {
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..N))], schema()),
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FILE,
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    create_index(&mut dataset, "val", &zone_map_params(64, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    ))
    .await;
    assert!(!reports[0].masked);
    assert_eq!(reports[0].scanned_fragments, vec![0, 1]);
    assert!(reports[0].partial_fragments.is_empty());
    let control_dir = TempStrDir::default();
    let mut control = unseeded_dataset(control_dir.as_str(), 1).await;
    create_index(
        &mut control,
        "val",
        &zone_map_params(1000, true, true),
        None,
    )
    .await;
    assert_eq!(
        layout(&zone_summaries(&dataset, "val").await),
        layout(&zone_summaries(&control, "val").await)
    );
    // Legacy storage is not planned through the zone map; results still
    // equal the scan.
    for filter in VAL_FILTERS {
        assert_eq!(
            ids_matching(&dataset, filter, true, false).await,
            ids_matching(&dataset, filter, false, false).await,
            "{filter}"
        );
    }
}

/// Stable row ids change nothing: selection and attribution use physical
/// row addresses.
#[tokio::test]
async fn test_stable_row_ids_reuse_by_physical_address() {
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..N))], schema()),
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FILE,
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    create_index(&mut dataset, "val", &zone_map_params(64, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    append(&mut dataset, 2 * N..3 * N).await;
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    ))
    .await;
    assert_eq!(reports[0].partial_fragments, vec![1, 2]);
    let control_dir = TempStrDir::default();
    let mut control = unseeded_dataset(control_dir.as_str(), 2).await;
    create_index(
        &mut control,
        "val",
        &zone_map_params(1000, true, true),
        None,
    )
    .await;
    assert_eq!(
        zone_summaries(&dataset, "val").await,
        zone_summaries(&control, "val").await
    );
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
}

/// The null positions a seed carries, if any.
async fn seed_positions(dataset: &Dataset, fragment_id: u64, column: &str) -> Option<Vec<u32>> {
    use arrow_ipc::reader::FileReader as IpcReader;
    let fragment = dataset
        .fragments()
        .iter()
        .find(|f| f.id == fragment_id)
        .unwrap()
        .clone();
    let reader = open_data_file(dataset, fragment.files.last().unwrap()).await;
    let value = reader
        .metadata()
        .file_schema
        .metadata
        .get(&format!("{SEED_META_KEY_PREFIX}{column}"))?
        .clone();
    let buf_index: u32 = value.split(':').next().unwrap().parse().unwrap();
    let bytes = reader.read_global_buffer(buf_index).await.unwrap();
    let ipc = IpcReader::try_new(std::io::Cursor::new(bytes.as_ref()), None).unwrap();
    let hex = ipc.schema().metadata.get("seed_null_bitmap")?.clone();
    let raw: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    if raw.is_empty() {
        return Some(Vec::new());
    }
    Some(
        RoaringBitmap::deserialize_from(raw.as_slice())
            .unwrap()
            .iter()
            .collect(),
    )
}

/// New seeds follow the index: its zone size as the record size, and its
/// null bitmap policy; an index created without a bitmap writes counts
/// only, and a later bitmap-keeping index scans such fragments.
#[tokio::test]
async fn test_seeds_follow_the_index_configuration() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, false),
        None,
    )
    .await;
    append(&mut dataset, N..2 * N).await;
    assert_eq!(
        seed_locator(&dataset, 1, "val")
            .await
            .as_deref()
            .unwrap()
            .split(':')
            .nth(1),
        Some("1000")
    );
    assert!(seed_positions(&dataset, 1, "val").await.is_none());
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(details(&segments[0]).has_null_bitmap, Some(false));

    // The bitmap-less index reuses the seed whole on update.
    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    })
    .await;
    assert_eq!(reports[0].reused_fragments, vec![1]);
    assert_eq!(reports[0].scanned_rows, 0);
    assert_index_matches_scan_with(&dataset, VAL_FILTERS, false).await;

    // A bitmap-keeping index of the same size must scan that fragment for
    // its positions, and its own appends then carry positions.
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    )
    .await;
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    ))
    .await;
    assert_eq!(reports[0].scanned_fragments, vec![0, 1]);
    append(&mut dataset, 2 * N..3 * N).await;
    let positions = seed_positions(&dataset, 2, "val").await.unwrap();
    assert_eq!(
        positions.len(),
        (0..N).filter(|i| val(2 * N + i).is_none()).count()
    );
    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    })
    .await;
    assert_eq!(reports[0].reused_fragments, vec![2]);
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
    let segments = index_metadata(&dataset, "val").await;
    let index = dataset
        .open_scalar_index("val", &segments[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        index
            .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
            .await
            .unwrap()
            .is_exact()
    );
}

/// Two zone maps on one column share one seed at the finer size with
/// positions if either keeps them; each index still cuts its own zones.
#[tokio::test]
async fn test_two_zone_maps_on_one_column_share_the_finer_seed() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    CreateIndexBuilder::new(
        &mut dataset,
        &["val"],
        IndexType::ZoneMap,
        &zone_map_params(640, true, false),
    )
    .name("coarse".to_string())
    .await
    .unwrap();
    CreateIndexBuilder::new(
        &mut dataset,
        &["val"],
        IndexType::ZoneMap,
        &zone_map_params(64, true, true),
    )
    .name("fine".to_string())
    .await
    .unwrap();
    append(&mut dataset, N..2 * N).await;
    assert_eq!(
        seed_locator(&dataset, 1, "val")
            .await
            .as_deref()
            .unwrap()
            .split(':')
            .nth(1),
        Some("64")
    );
    assert!(seed_positions(&dataset, 1, "val").await.is_some());
    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    })
    .await;
    // Both indexes updated from the shared seed, no scan.
    assert_eq!(reports.len(), 2);
    for report in &reports {
        assert_eq!(report.reused_fragments, vec![1]);
        assert_eq!(report.scanned_rows, 0);
    }
    for name in ["coarse", "fine"] {
        let segments: Vec<IndexMetadata> = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .filter(|i| i.name == name)
            .cloned()
            .collect();
        assert_eq!(segments.len(), 1);
        let index = dataset
            .open_scalar_index("val", &segments[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let zm = index.as_any().downcast_ref::<ZoneMapIndex>().unwrap();
        let t = if name == "coarse" { 640 } else { 64 };
        assert_eq!(
            layout(&zm.zone_summaries()),
            (0..2u64)
                .flat_map(|f| target_layout(f, N as u64, t))
                .collect::<Vec<_>>()
        );
    }
}

/// A rebuild over a subset (two segments merged) goes through the
/// metadata-aware builder: a scanned fragment keeps its configured
/// boundaries up to its deleted tail, a fully seed-covered deleted fragment
/// keeps its physical zones, fragments outside the request get no zones,
/// and the committed coverage is exactly the request.
#[tokio::test]
async fn test_rebuild_keeps_boundaries_past_a_deleted_tail() {
    let dir = TempStrDir::default();
    let mut dataset = seeded_dataset(dir.as_str(), 64, 1, true).await;
    // Fragment 1 (64-row seeds, crossing a 1,000-row boundary) loses its
    // physical tail, so the rebuild scans it whole; fragment 2 is appended
    // under the 1,000-row index, so its seed fits and it is reused whole.
    dataset
        .delete(&format!("id >= {} AND id < {}", 2 * N - 1000, 2 * N))
        .await
        .unwrap();
    // Three segments: fragment 0 (replace), fragment 1 (append mode), then
    // fragment 2 appended and indexed in append mode; merging the trailing
    // two is a rebuild over {1, 2} with fragment 0 outside the request.
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        Some(vec![0]),
    )
    .await;
    dataset
        .optimize_indices(&OptimizeOptions::new().num_indices_to_merge(Some(0)))
        .await
        .unwrap();
    append(&mut dataset, 2 * N..3 * N).await;
    dataset
        .optimize_indices(&OptimizeOptions::new().num_indices_to_merge(Some(0)))
        .await
        .unwrap();
    assert_eq!(index_metadata(&dataset, "val").await.len(), 3);
    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::new().num_indices_to_merge(Some(2)))
            .await
            .unwrap();
    })
    .await;
    let report = reports.last().unwrap();
    assert_eq!(
        (
            report.partial_fragments.clone(),
            report.scanned_fragments.clone(),
            report.reused_fragments.clone(),
            report.masked,
        ),
        (vec![], vec![1], vec![2], true),
        "{report:?}"
    );
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(segments.len(), 2);
    let merged = segments
        .iter()
        .find(|s| coverage(s) == RoaringBitmap::from_iter([1u32, 2]))
        .expect("the rebuilt segment covers exactly {1, 2}");
    let index = dataset
        .open_scalar_index("val", &merged.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let mut zones = index
        .as_any()
        .downcast_ref::<ZoneMapIndex>()
        .unwrap()
        .zone_summaries();
    zones.sort_by_key(|z| (z.bound.fragment_id, z.bound.start));
    let mut expected: Vec<(u64, u64, usize)> = (1..3u64)
        .flat_map(|f| target_layout(f, N as u64, 1000))
        .collect();
    assert!(zones.iter().all(|z| z.bound.fragment_id != 0));
    // Fragment 1 is scanned whole (a deleted fragment with crossing records):
    // its zones keep the configured boundaries, and its last zone, whose
    // rows [19000, 20000) are all deleted, is not emitted. Fragment 2 keeps
    // every zone from its seed.
    expected.retain(|(f, start, _)| !(*f == 1 && *start == 19_000));
    assert_eq!(layout(&zones), expected);
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
}

/// Floating-point seeds with NaN of both signs and infinities fold into
/// large zones and keep index results equal to a scan. This test uses only
/// public APIs so the same assertions run on the base revision.
#[tokio::test]
async fn test_float_nan_queries_match_scan_after_seed_reuse() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(&mut dataset, "x", &zone_map_params(8, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    append(&mut dataset, 2 * N..3 * N).await;
    dataset.drop_index(&index_name("x")).await.unwrap();
    create_index(&mut dataset, "x", &zone_map_params(20000, true, true), None).await;
    assert_index_matches_scan(
        &dataset,
        &[
            "x < 0",
            "x > 0",
            "x >= 1.5",
            "x <= -40",
            "x = 2",
            "x = 48",
            "x BETWEEN -1 AND 1",
            "x IN (3, 5, 1e308)",
            "x IS NULL",
            "x IS NOT NULL",
            "x > 1e308",
            "x < -1e308",
        ],
    )
    .await;
}

/// A current overlay on the indexed column changes values the base file's
/// seed never saw, so that fragment is scanned, at creation and on update.
/// An overlay on another column leaves the seed usable.
#[tokio::test]
async fn test_overlaid_fragment_is_scanned_instead_of_reused() {
    use super::dataset_overlay_index_masking::commit_overlay;
    use lance_table::format::overlay::OverlayCoverage;

    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(&mut dataset, "val", &zone_map_params(8, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    append(&mut dataset, 2 * N..3 * N).await;
    let val_field = dataset.schema().field("val").unwrap().id;
    let name_field = dataset.schema().field("name").unwrap().id;
    // Fragment 1, offset 0 (id = N): val becomes 10,000,000, beyond every seed bound.
    let dataset = commit_overlay(
        dataset,
        "val_overlay",
        1,
        &[val_field],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0])),
        vec![Arc::new(Int64Array::from(vec![Some(10_000_000)]))],
    )
    .await;
    // Fragment 2 gets an overlay on an unrelated column.
    let mut dataset = commit_overlay(
        dataset,
        "name_overlay",
        2,
        &[name_field],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0])),
        vec![Arc::new(StringArray::from(vec![Some("overlaid")]))],
    )
    .await;

    // Creation: fragment 1 is scanned, fragment 2 keeps using its seed.
    dataset.drop_index(&index_name("val")).await.unwrap();
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(N as u64, true, true),
        None,
    ))
    .await;
    assert_eq!(reports[0].reused_fragments, vec![2]);
    assert_eq!(reports[0].scanned_fragments, vec![0, 1]);
    assert_eq!(
        ids_matching(&dataset, "val = 10000000", true, true).await,
        BTreeSet::from([N])
    );
    assert_eq!(
        ids_matching(&dataset, "val > 9000000", true, true).await,
        BTreeSet::from([N])
    );
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;

    // Update: the index covers fragment 0 only; maintenance scans fragment 1
    // and reuses fragment 2.
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(N as u64, true, true),
        Some(vec![0]),
    )
    .await;
    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    })
    .await;
    assert_eq!(reports[0].reused_fragments, vec![2]);
    assert_eq!(reports[0].scanned_fragments, vec![1]);
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(coverage(&segments[0]), RoaringBitmap::from_iter(0..3u32));
    assert_eq!(
        ids_matching(&dataset, "val = 10000000", true, true).await,
        BTreeSet::from([N])
    );
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
}

/// Historical availability is not configuration: an index whose details say
/// `has_null_bitmap = false` but whose file carries no explicit
/// `store_null_bitmap = false` (a legacy segment, or one updated or merged
/// from a legacy segment) keeps the default policy, so the seeds written
/// for it carry null positions.
#[tokio::test]
async fn test_historical_index_without_bitmap_still_collects_seed_positions() {
    use std::collections::HashMap;

    use arrow_array::{UInt32Array, UInt64Array};
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_index::scalar::zonemap::ZONEMAP_INDEX_FILENAME;
    use lance_index::scalar::{IndexStore, index_files_to_table};
    use uuid::Uuid;

    use crate::dataset::index::LanceIndexStoreExt;
    use crate::index::scalar::zonemap::stored_null_bitmap_policy;

    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;

    // Hand-write a legacy-style segment: one zone over fragment 0, no null
    // bitmap buffer, no policy key, details with `has_null_bitmap: false`
    // and seeds enabled.
    let values: Vec<i64> = (0..N).filter_map(val).collect();
    let null_count = (0..N).filter(|i| val(*i).is_none()).count() as u32;
    let uuid = Uuid::new_v4();
    let store = LanceIndexStore::from_dataset_for_new(&dataset, &uuid).unwrap();
    let mut file_schema = ArrowSchema::new(vec![
        ArrowField::new("min", DataType::Int64, true),
        ArrowField::new("max", DataType::Int64, true),
        ArrowField::new("null_count", DataType::UInt32, false),
        ArrowField::new("nan_count", DataType::UInt32, false),
        ArrowField::new("fragment_id", DataType::UInt64, false),
        ArrowField::new("zone_start", DataType::UInt64, false),
        ArrowField::new("zone_length", DataType::UInt64, false),
    ]);
    file_schema
        .metadata
        .insert("rows_per_zone".to_string(), N.to_string());
    let file_schema = Arc::new(file_schema);
    let batch = RecordBatch::try_new(
        file_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![*values.iter().min().unwrap()])),
            Arc::new(Int64Array::from(vec![*values.iter().max().unwrap()])),
            Arc::new(UInt32Array::from(vec![null_count])),
            Arc::new(UInt32Array::from(vec![0u32])),
            Arc::new(UInt64Array::from(vec![0u64])),
            Arc::new(UInt64Array::from(vec![0u64])),
            Arc::new(UInt64Array::from(vec![N as u64])),
        ],
    )
    .unwrap();
    let mut file = store
        .new_index_file(ZONEMAP_INDEX_FILENAME, file_schema)
        .await
        .unwrap();
    file.write_record_batch(batch).await.unwrap();
    let index_file = file.finish_with_metadata(HashMap::new()).await.unwrap();
    let legacy_details = prost_types::Any::from_msg(&pbold::ZoneMapIndexDetails {
        rows_per_zone: Some(N as u64),
        use_seeds: Some(true),
        has_null_bitmap: Some(false),
    })
    .unwrap();
    let segment = IndexMetadata {
        uuid,
        fields: vec![dataset.schema().field("val").unwrap().id],
        covering_fields: vec![],
        name: index_name("val"),
        dataset_version: dataset.version().version,
        fragment_bitmap: Some(RoaringBitmap::from_iter([0u32])),
        index_details: Some(Arc::new(legacy_details)),
        index_version: 0,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: Some(index_files_to_table(vec![index_file])),
        clustering_id: None,
        metadata: Default::default(),
    };
    dataset
        .commit_existing_index_segments(&index_name("val"), "val", vec![segment])
        .await
        .unwrap();
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(details(&segments[0]).has_null_bitmap, Some(false));
    assert!(
        stored_null_bitmap_policy(&dataset, &segments[0])
            .await
            .unwrap()
    );

    // The appended fragment's seed carries positions: availability did not
    // turn the default off.
    append(&mut dataset, N..2 * N).await;
    let positions = seed_positions(&dataset, 1, "val").await.unwrap();
    assert_eq!(
        positions.len(),
        (0..N).filter(|i| val(N + i).is_none()).count()
    );

    // Maintenance rebuilds the version 0 segment as version 1 from source
    // data: fragment 1 comes from its seed (positions included), fragment 0
    // is scanned, and the default policy regains the bitmap.
    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    })
    .await;
    assert_eq!(reports[0].reused_fragments, vec![1]);
    assert_eq!(reports[0].scanned_fragments, vec![0]);
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].index_version, 1);
    assert_eq!(details(&segments[0]).has_null_bitmap, Some(true));
    assert_index_matches_scan_with(&dataset, VAL_FILTERS, true).await;
}

/// The policy lookup costs one index-file footer read per ZoneMap segment
/// on a cold handle and none once the file metadata is cached; several
/// segments of one index still produce one seed writer.
#[tokio::test]
async fn test_seed_collector_policy_lookup_cost() {
    use lance_file::version::ConcreteFileVersion;

    use crate::dataset::write::seeds::SeedCollector;

    async fn index_requests(dataset: &Dataset) -> usize {
        io(dataset)
            .requests
            .iter()
            .filter(|r| r.path.to_string().contains("_indices"))
            .count()
    }
    async fn writers_and_requests(dataset: &Dataset) -> (usize, usize) {
        io(dataset);
        let collector = SeedCollector::for_write(
            ConcreteFileVersion::V2_0,
            Some(dataset),
            dataset.schema(),
            true,
        )
        .await
        .unwrap();
        (collector.len(), index_requests(dataset).await)
    }

    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    )
    .await;

    // Cold handle: one segment, one footer read; warm: none.
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    let (writers, cold) = writers_and_requests(&dataset).await;
    assert_eq!(writers, 1);
    assert!(
        (1..=2).contains(&cold),
        "cold lookup read the index file {cold} times"
    );
    let (writers, warm) = writers_and_requests(&dataset).await;
    assert_eq!((writers, warm), (1, 0));

    // Two segments of the same index: one lookup each, still one writer.
    let mut dataset = dataset;
    append(&mut dataset, N..2 * N).await;
    dataset
        .optimize_indices(&OptimizeOptions::new().num_indices_to_merge(Some(0)))
        .await
        .unwrap();
    assert_eq!(index_metadata(&dataset, "val").await.len(), 2);
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    let (writers, cold) = writers_and_requests(&dataset).await;
    assert_eq!(writers, 1);
    assert!(
        (2..=4).contains(&cold),
        "two segments read {cold} times cold"
    );
    let (writers, warm) = writers_and_requests(&dataset).await;
    assert_eq!((writers, warm), (1, 0));
}

/// A segment whose index file cannot be read costs the write only that
/// index's seed, never the write itself; a healthy index on another column
/// keeps its seed in the same file.
#[tokio::test]
async fn test_unreadable_policy_skips_the_seed_not_the_write() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    )
    .await;
    create_index(
        &mut dataset,
        "name",
        &zone_map_params(1000, true, true),
        None,
    )
    .await;
    let segment = index_metadata(&dataset, "val").await.remove(0);
    std::fs::remove_file(format!(
        "{}/_indices/{}/zonemap.lance",
        dir.as_str(),
        segment.uuid
    ))
    .unwrap();

    // A fresh handle: nothing about the removed file is cached.
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    append(&mut dataset, N..2 * N).await;
    let fragment = dataset.fragments().last().unwrap().id;
    assert!(
        seed_locator(&dataset, fragment, "val").await.is_none(),
        "the unreadable index gets no seed"
    );
    assert!(
        seed_locator(&dataset, fragment, "name").await.is_some(),
        "the healthy index on another column keeps its seed"
    );

    // A second, healthy zone map on the same column still gets its seed:
    // only the failed consumer is skipped, not the column.
    CreateIndexBuilder::new(
        &mut dataset,
        &["val"],
        IndexType::ZoneMap,
        &zone_map_params(500, true, true),
    )
    .name("val_fine".to_string())
    .await
    .unwrap();
    let dataset_fresh = Dataset::open(dir.as_str()).await.unwrap();
    let mut dataset = dataset_fresh;
    append(&mut dataset, 2 * N..3 * N).await;
    let fragment = dataset.fragments().last().unwrap().id;
    let locator = seed_locator(&dataset, fragment, "val").await;
    assert_eq!(
        locator
            .as_deref()
            .and_then(|locator| locator.split_once(':'))
            .map(|(_, rows)| rows),
        Some("500"),
        "the healthy same-column index drives the seed: {locator:?}"
    );
}

/// Counts overlapping reads and slows each one down enough for the overlap
/// to show.
#[derive(Debug, Default)]
struct ReadWatch {
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl ReadWatch {
    async fn enter(&self) -> ReadGuard<'_> {
        let live = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(live, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(5)).await;
        ReadGuard(self)
    }
}

struct ReadGuard<'a>(&'a ReadWatch);

impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct SlowReads {
    target: Arc<dyn object_store::ObjectStore>,
    watch: Arc<ReadWatch>,
}

impl std::fmt::Display for SlowReads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SlowReads({})", self.target)
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for SlowReads {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        bytes: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.target.put_opts(location, bytes, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.target.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        let _read = self.watch.enter().await;
        self.target.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        let _read = self.watch.enter().await;
        self.target.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.target.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.target.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        opts: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.target.copy_opts(from, to, opts).await
    }

    async fn rename_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        opts: object_store::RenameOptions,
    ) -> object_store::Result<()> {
        self.target.rename_opts(from, to, opts).await
    }
}

#[derive(Debug)]
struct SlowReadsWrapper(Arc<ReadWatch>);

impl lance_io::object_store::WrappingObjectStore for SlowReadsWrapper {
    fn wrap(
        &self,
        _store_prefix: &str,
        target: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(SlowReads {
            target,
            watch: self.0.clone(),
        })
    }

    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        Some(original)
    }
}

/// Seeds are harvested a bounded number of fragments at a time: the footer
/// reads overlap, never beyond the harvest concurrency, and a raw seed
/// buffer is released as soon as its fragment is decided, so no more than
/// one is held at a time.
#[tokio::test]
#[allow(clippy::print_stderr)]
async fn test_harvest_reads_are_bounded_and_concurrent() {
    let watch = Arc::new(ReadWatch::default());
    let store_params = lance_io::object_store::ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(SlowReadsWrapper(watch.clone()))),
        ..Default::default()
    };
    // An in-memory store, so data-file reads go through the `object_store`
    // layer the wrapper counts (local files are read directly), and a
    // session without a metadata cache, so every harvest reads its footer.
    let session = Arc::new(crate::session::Session::new(
        0,
        0,
        Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
    ));
    let params = WriteParams {
        max_rows_per_file: ROWS_PER_FILE,
        store_params: Some(store_params),
        session: Some(session),
        ..Default::default()
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..1000))], schema()),
        "memory://zonemap-harvest-bound",
        Some(params),
    )
    .await
    .unwrap();
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    )
    .await;
    let fragments = 3 * harvest_concurrency(&dataset) as i32;
    for k in 1..=fragments {
        append(&mut dataset, k * 1000..(k + 1) * 1000).await;
    }
    let seeded: Vec<u32> = (1..=fragments as u32).collect();

    watch.max_in_flight.store(0, Ordering::SeqCst);
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        Some(seeded.clone()),
    ))
    .await;
    assert_eq!(reports.len(), 1);
    let report = &reports[0];
    // The task bound: harvests in flight never exceed the configured
    // concurrency, and the window is actually used.
    let concurrency = harvest_concurrency(&dataset);
    assert_eq!(report.harvest_concurrency, concurrency);
    assert!(
        (2..=concurrency).contains(&report.max_concurrent_harvests),
        "{} concurrent harvests for a concurrency of {concurrency}",
        report.max_concurrent_harvests
    );
    // Storage-level concurrency is reported separately: one harvest can
    // issue several reads, so only the overlap itself is asserted.
    let max_reads_in_flight = watch.max_in_flight.load(Ordering::SeqCst);
    assert!(
        max_reads_in_flight >= 2,
        "{max_reads_in_flight} overlapping storage reads"
    );
    // Visible with `--nocapture`: the measured bounds for the checkpoint.
    eprintln!(
        "harvest concurrency {concurrency}: {} concurrent harvests, {max_reads_in_flight} \
         concurrent storage reads, {} live seed bytes over {} total",
        report.max_concurrent_harvests, report.max_live_seed_bytes, report.total_seed_bytes
    );
    assert_eq!(
        report.reused_fragments,
        seeded.iter().map(|id| u64::from(*id)).collect::<Vec<_>>(),
        "every fragment is folded from its seed, so no scan read is counted"
    );
    assert_eq!(report.scanned_rows, 0);
    assert_eq!(report.scans, 0, "nothing left to scan, no scan issued");
    let per_seed = report.total_seed_bytes / seeded.len();
    assert!(
        report.max_live_seed_bytes <= per_seed,
        "{} live seed bytes, {per_seed} per seed",
        report.max_live_seed_bytes
    );
    assert_index_matches_scan(&dataset, &["val = 23", "val IS NULL", "val > 500000"]).await;
}

/// Null positions survive a binary copy only when every input carried them;
/// counts always survive. A bitmap-keeping index built over an output
/// without positions scans it, a count-only index ignores carried positions
/// and keeps no bitmap, and enabling collection later cannot recover
/// positions the inputs never had.
#[rstest::rstest]
#[case::positions_on_on(true, true, None)]
#[case::positions_on_off(true, false, None)]
#[case::positions_off_off(false, false, None)]
#[case::positions_off_on(false, true, None)]
#[case::count_only_after_copy(true, true, Some(false))]
#[case::bitmap_enabled_after_copy(false, false, Some(true))]
#[tokio::test]
async fn test_binary_copy_bitmap_provenance(
    #[case] first: bool,
    #[case] second: bool,
    #[case] after_copy: Option<bool>,
) {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, first),
        None,
    )
    .await;
    append(&mut dataset, N..2 * N).await;
    // The replacement keeps covering only the pre-index fragment, so both
    // appended fragments stay uncovered and the planner bins them together.
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, second),
        Some(vec![0]),
    )
    .await;
    append(&mut dataset, 2 * N..3 * N).await;
    assert_eq!(seed_positions(&dataset, 1, "val").await.is_some(), first);
    assert_eq!(seed_positions(&dataset, 2, "val").await.is_some(), second);

    // Only the two seeded inputs go into the copy.
    dataset.delete(&format!("id < {N}")).await.unwrap();
    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 10 * ROWS_PER_FILE,
            compaction_mode: Some(CompactionMode::ForceBinaryCopy),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_added, 1);
    let output = dataset.fragments().iter().map(|f| f.id).max().unwrap();
    assert!(seed_locator(&dataset, output, "val").await.is_some());
    let positions = seed_positions(&dataset, output, "val").await;
    assert_eq!(positions.is_some(), first && second);
    if let Some(positions) = &positions {
        let expected: Vec<u32> = (N..3 * N)
            .filter(|i| val(*i).is_none())
            .map(|i| (i - N) as u32)
            .collect();
        assert_eq!(positions, &expected);
    }

    let policy = after_copy.unwrap_or(second);
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, policy),
        None,
    ))
    .await;
    assert_eq!(reports.len(), 1);
    if policy && positions.is_none() {
        assert_eq!(
            reports[0].scanned_fragments,
            vec![output],
            "no positions to reuse"
        );
        assert!(reports[0].reused_fragments.is_empty());
    } else {
        assert_eq!(reports[0].reused_fragments, vec![output]);
        assert_eq!(reports[0].scanned_rows, 0);
    }
    let segment = index_metadata(&dataset, "val").await.remove(0);
    assert_eq!(details(&segment).has_null_bitmap, Some(policy));
    assert_index_matches_scan_with(
        &dataset,
        &["val IS NULL", "val IS NOT NULL", "val = 23"],
        policy,
    )
    .await;

    // The next seeded append and update follow the index's policy.
    append(&mut dataset, 3 * N..4 * N).await;
    let appended = dataset.fragments().last().unwrap().id;
    assert_eq!(
        seed_positions(&dataset, appended, "val").await.is_some(),
        policy
    );
    dataset
        .optimize_indices(&OptimizeOptions::new().num_indices_to_merge(Some(0)))
        .await
        .unwrap();
    for segment in index_metadata(&dataset, "val").await {
        assert_eq!(details(&segment).has_null_bitmap, Some(policy));
    }
    assert_index_matches_scan_with(
        &dataset,
        &["val IS NULL", "val IS NOT NULL", "val = 23"],
        policy,
    )
    .await;
}
