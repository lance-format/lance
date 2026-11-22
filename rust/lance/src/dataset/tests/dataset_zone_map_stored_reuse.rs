// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The ordinary zone map build takes a fragment's statistics whole from the
//! stored zones of a committed segment of the same index when exactly one
//! segment offers them and they are exactly the fragment's target zones;
//! otherwise from the fragment's write seed, otherwise from a scan. What was
//! taken from where is read from the build's own report, data reads from
//! the object store's request log, and every query is compared with a plain
//! scan by row id.

use std::collections::{BTreeSet, HashSet};
use std::sync::{Arc, Mutex};

use arrow_array::{
    BinaryArray, Float64Array, Int8Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_index::IndexType;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::{TrainingCriteria, TrainingOrdering, VALUE_COLUMN_NAME};
use lance_index::scalar::zonemap::{
    StoredZonesLoad, ZoneMapIndex, ZoneMapIndexBuilder, ZoneMapIndexBuilderParams,
    ZoneMapIndexVersion,
};
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams, index_files_to_table};
use lance_io::utils::tracking_store::IoStats;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::cleanup::CleanupPolicyBuilder;
use crate::dataset::index::LanceIndexStoreExt;
use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::{WriteMode, WriteParams};
use crate::index::scalar::zonemap::{
    SEED_REUSE_REPORTS, SeedReuseReport, TEST_STORED_ZONE_ADMISSION_BYTES,
};
use crate::index::{CreateIndexBuilder, DatasetIndexExt};

/// Rows per written fragment.
const ROWS: i32 = 1000;
/// Zones of 300 rows: three full zones and a short last one per fragment.
const T: u64 = 300;

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("val", DataType::Int64, true),
        ArrowField::new("x", DataType::Float64, true),
        ArrowField::new("name", DataType::Utf8, true),
        ArrowField::new("bin", DataType::Binary, true),
        ArrowField::new("small", DataType::Int8, true),
    ]))
}

/// Null every fifth row, otherwise scrambled so zones overlap.
fn val(i: i32) -> Option<i64> {
    (i % 5 != 0).then(|| i64::from(i) * 7 % 1013 + i64::from(i) / 100 * 1000)
}

fn x(i: i32) -> Option<f64> {
    match i % 13 {
        0 => None,
        1 => Some(f64::NAN),
        2 => Some(-f64::NAN),
        3 => Some(f64::INFINITY),
        _ => Some(f64::from(i % 97) - 48.0),
    }
}

fn name(i: i32) -> Option<String> {
    (i % 11 != 0).then(|| format!("n{:05}", (i * 37) % 20000))
}

fn bin(i: i32) -> Option<Vec<u8>> {
    (i % 7 != 0).then(|| vec![(i % 251) as u8, (i / 251 % 251) as u8])
}

fn small(i: i32) -> Option<i8> {
    (i % 9 != 0).then(|| (i % 200 - 100) as i8)
}

fn rows(ids: std::ops::Range<i32>) -> RecordBatch {
    let bins: Vec<Option<Vec<u8>>> = ids.clone().map(bin).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(ids.clone())),
            Arc::new(Int64Array::from_iter(ids.clone().map(val))),
            Arc::new(Float64Array::from_iter(ids.clone().map(x))),
            Arc::new(StringArray::from_iter(ids.clone().map(name))),
            Arc::new(BinaryArray::from_iter(bins.iter().map(|b| b.as_deref()))),
            Arc::new(Int8Array::from_iter(ids.map(small))),
        ],
    )
    .unwrap()
}

fn write_params(stable_row_ids: bool) -> WriteParams {
    WriteParams {
        max_rows_per_file: ROWS as usize,
        enable_stable_row_ids: stable_row_ids,
        ..Default::default()
    }
}

/// `fragments` fragments of [`ROWS`] rows each.
async fn write(uri: &str, fragments: i32) -> Dataset {
    write_with(uri, fragments, false).await
}

async fn write_with(uri: &str, fragments: i32, stable_row_ids: bool) -> Dataset {
    Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..fragments * ROWS))], schema()),
        uri,
        Some(write_params(stable_row_ids)),
    )
    .await
    .unwrap()
}

/// Append one fragment of `count` rows starting at `start`.
async fn append(dataset: &mut Dataset, start: i32, count: i32) {
    dataset
        .append(
            RecordBatchIterator::new([Ok(rows(start..start + count))], schema()),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..write_params(false)
            }),
        )
        .await
        .unwrap();
}

fn index_name(column: &str) -> String {
    format!("{column}_zones")
}

fn params(rows_per_zone: u64, use_seeds: bool, store_null_bitmap: bool) -> ScalarIndexParams {
    ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(&serde_json::json!({
        "rows_per_zone": rows_per_zone,
        "use_seeds": use_seeds,
        "store_null_bitmap": store_null_bitmap,
    }))
}

fn default_params() -> ScalarIndexParams {
    params(T, false, true)
}

/// Create or replace the index named `name` on `column` over every fragment.
async fn build_named(dataset: &mut Dataset, column: &str, name: &str, params: &ScalarIndexParams) {
    CreateIndexBuilder::new(dataset, &[column], IndexType::ZoneMap, params)
        .name(name.to_string())
        .replace(true)
        .await
        .unwrap();
}

async fn build(dataset: &mut Dataset, column: &str, params: &ScalarIndexParams) {
    build_named(dataset, column, &index_name(column), params).await;
}

/// Run `future` and return the zone map build reports it produced.
async fn with_reports<F, R>(future: F) -> (R, Vec<SeedReuseReport>)
where
    F: std::future::Future<Output = R>,
{
    SEED_REUSE_REPORTS
        .scope(Mutex::new(Vec::new()), async {
            let out = future.await;
            let reports = SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone());
            (out, reports)
        })
        .await
}

/// Rebuild the index on `column` over every fragment and return the build's
/// only report and the object store requests it made.
async fn rebuild(
    dataset: &mut Dataset,
    column: &str,
    params: &ScalarIndexParams,
) -> (SeedReuseReport, IoStats) {
    io(dataset);
    let (_, reports) = with_reports(build(dataset, column, params)).await;
    let io = io(dataset);
    assert_eq!(reports.len(), 1, "{reports:?}");
    (reports.into_iter().next().unwrap(), io)
}

fn io(dataset: &Dataset) -> IoStats {
    dataset.object_store.io_stats_incremental()
}

async fn segments(dataset: &Dataset, column: &str) -> Vec<IndexMetadata> {
    dataset
        .load_indices_by_name(&index_name(column))
        .await
        .unwrap()
}

fn fragment_ids(dataset: &Dataset) -> Vec<u64> {
    dataset.fragments().iter().map(|f| f.id).collect()
}

/// Data files serving the fragments `ids`.
fn data_files(dataset: &Dataset, ids: &[u64]) -> HashSet<String> {
    dataset
        .fragments()
        .iter()
        .filter(|fragment| ids.contains(&fragment.id))
        .flat_map(|fragment| fragment.files.iter().map(|file| file.path.clone()))
        .collect()
}

/// Read requests against the data files serving `ids`.
fn data_reads(dataset: &Dataset, io: &IoStats, ids: &[u64]) -> Vec<String> {
    let files = data_files(dataset, ids);
    io.requests
        .iter()
        .filter(|request| request.method.starts_with("get"))
        .map(|request| request.path.to_string())
        .filter(|path| {
            !path.contains("_indices/") && files.iter().any(|file| path.ends_with(file.as_str()))
        })
        .collect()
}

/// Read requests against the index directory of `segment`.
fn segment_reads(io: &IoStats, segment: &Uuid) -> usize {
    let dir = format!("_indices/{segment}");
    io.requests
        .iter()
        .filter(|request| request.path.to_string().contains(&dir))
        .count()
}

/// Row ids matching `filter`, through the index or through a plain scan.
async fn ids_matching(dataset: &Dataset, filter: &str, use_index: bool) -> BTreeSet<u64> {
    let mut scanner = dataset.scan();
    scanner
        .filter(filter)
        .unwrap()
        .project(&["id"])
        .unwrap()
        .with_row_id()
        .use_scalar_index(use_index);
    let batch = scanner.try_into_batch().await.unwrap();
    batch[ROW_ID]
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .values()
        .iter()
        .copied()
        .collect()
}

/// Each filter yields the same rows through the index and through a scan;
/// the ones in `indexed` must be planned with the zone map.
async fn assert_matches_scan(dataset: &Dataset, filters: &[&str], indexed: &[&str]) {
    for filter in filters {
        let through_index = ids_matching(dataset, filter, true).await;
        let scanned = ids_matching(dataset, filter, false).await;
        assert_eq!(through_index, scanned, "{filter}");
    }
    for filter in indexed {
        let mut scanner = dataset.scan();
        scanner.filter(filter).unwrap();
        let plan = scanner.explain_plan(true).await.unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "{filter}:\n{plan}");
    }
}

const VAL_FILTERS: &[&str] = &[
    "val > 500",
    "val < 2300",
    "val = 707",
    "val BETWEEN 100 AND 110",
    "val IS NULL",
    "val IS NOT NULL",
];

fn filters(column: &str) -> (&'static [&'static str], &'static [&'static str]) {
    match column {
        "val" => (VAL_FILTERS, &["val > 500", "val = 707", "val IS NULL"]),
        "x" => (
            &[
                "x > 10",
                "x < -40",
                "x = 3",
                "x >= 0 AND x <= 1",
                "x IS NULL",
                "x IS NOT NULL",
            ],
            &["x > 10", "x = 3", "x IS NULL"],
        ),
        "name" => (
            &[
                "name > 'n15000'",
                "name < 'n00500'",
                "name = 'n00037'",
                "name IS NULL",
                "name IS NOT NULL",
            ],
            &["name > 'n15000'", "name = 'n00037'", "name IS NULL"],
        ),
        "bin" => (
            &["bin IS NULL", "bin IS NOT NULL", "bin = X'0a00'"],
            &["bin IS NULL"],
        ),
        "small" => (
            &[
                "small > 90",
                "small = 7",
                "small IS NULL",
                "small IS NOT NULL",
            ],
            &["small > 90", "small = 7", "small IS NULL"],
        ),
        other => panic!("no filters for {other}"),
    }
}

async fn assert_column_matches_scan(dataset: &Dataset, column: &str) {
    let (all, indexed) = filters(column);
    assert_matches_scan(dataset, all, indexed).await;
}

/// 1. A whole rebuild with nothing appended takes every fragment from the
///    committed segment and reads no table data, for every value type.
#[tokio::test]
async fn test_rebuild_reuses_stored_zones_without_data_reads() {
    for column in ["val", "x", "name", "bin", "small"] {
        let dir = TempStrDir::default();
        let mut dataset = write(dir.as_str(), 3).await;
        let ids = fragment_ids(&dataset);
        let (_, first) = with_reports(build(&mut dataset, column, &default_params())).await;
        assert_eq!(first[0].scanned_fragments, ids, "{column}");
        let before = segments(&dataset, column).await;
        assert_eq!(before.len(), 1);

        let (report, io) = rebuild(&mut dataset, column, &default_params()).await;
        assert_eq!(report.stored_fragments, ids, "{column}");
        assert_eq!(report.stored_sources, 1, "{column}");
        assert!(report.stored_bytes > 0 && report.stored_payload_bytes > 0);
        assert!(report.scanned_fragments.is_empty(), "{column}");
        assert_eq!(report.scanned_rows, 0, "{column}");
        assert_eq!(report.total_seed_bytes, 0, "{column}");
        assert_eq!(
            data_reads(&dataset, &io, &ids),
            Vec::<String>::new(),
            "{column}"
        );

        let after = segments(&dataset, column).await;
        assert_eq!(after.len(), 1);
        assert_ne!(after[0].uuid, before[0].uuid, "{column}: replaced");
        assert_column_matches_scan(&dataset, column).await;
    }
}

/// 2. Mixed: untouched fragments come from the committed segment, a
///    compaction destination from its write seed (string column with seeds)
///    or from a scan (narrow column without seeds); the old segment is
///    replaced and results equal a scan.
#[tokio::test]
async fn test_mixed_sources_after_deferred_compaction() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    // Two small fragments that compaction joins.
    append(&mut dataset, 2 * ROWS, ROWS / 4).await;
    append(&mut dataset, 2 * ROWS + ROWS / 4, ROWS / 4).await;
    build(&mut dataset, "name", &params(T, true, true)).await;
    build(&mut dataset, "small", &params(T, false, true)).await;
    let old_name = segments(&dataset, "name").await[0].uuid;
    let old_small = segments(&dataset, "small").await[0].uuid;

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: ROWS as usize,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let ids = fragment_ids(&dataset);
    assert_eq!(&ids[..2], &[0, 1]);
    let destination = ids[2..].to_vec();
    assert_eq!(destination.len(), 1, "{ids:?}");

    let (report, io) = rebuild(&mut dataset, "name", &params(T, true, true)).await;
    assert_eq!(report.stored_fragments, vec![0, 1]);
    assert_eq!(
        report.reused_fragments, destination,
        "destination from its seed"
    );
    assert!(report.scanned_fragments.is_empty());
    assert_eq!(data_reads(&dataset, &io, &[0, 1]), Vec::<String>::new());

    let (report, io) = rebuild(&mut dataset, "small", &params(T, false, true)).await;
    assert_eq!(report.stored_fragments, vec![0, 1]);
    assert!(report.reused_fragments.is_empty());
    assert_eq!(report.scanned_fragments, destination, "destination scanned");
    assert_eq!(data_reads(&dataset, &io, &[0, 1]), Vec::<String>::new());

    for (column, old) in [("name", old_name), ("small", old_small)] {
        let now = segments(&dataset, column).await;
        assert_eq!(now.len(), 1);
        assert_ne!(now[0].uuid, old);
        assert_eq!(
            now[0].fragment_bitmap.clone().unwrap(),
            RoaringBitmap::from_iter(ids.iter().map(|id| *id as u32))
        );
        assert_column_matches_scan(&dataset, column).await;
    }
}

/// 3. Deferred compaction then a whole rebuild: untouched fragments are
///    reused, the destination is recomputed, the old segment is gone, and
///    the ordinary trim then releases the reuse version the old segment held.
#[tokio::test]
async fn test_rebuild_after_deferred_compaction_releases_the_reuse_version() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    append(&mut dataset, 2 * ROWS, ROWS / 4).await;
    append(&mut dataset, 2 * ROWS + ROWS / 4, ROWS / 4).await;
    build(&mut dataset, "val", &default_params()).await;
    let old = segments(&dataset, "val").await[0].uuid;
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: ROWS as usize,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let destination = fragment_ids(&dataset)[2..].to_vec();
    // The old segment still needs the reuse version.
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    let history = dataset.frag_reuse_index().await.unwrap().unwrap();
    assert_eq!(history.details.versions.len(), 1);

    let (report, _) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_eq!(report.stored_fragments, vec![0, 1]);
    assert_eq!(report.scanned_fragments, destination);
    let now = segments(&dataset, "val").await;
    assert_eq!(now.len(), 1);
    assert_ne!(now[0].uuid, old);
    let all = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    assert!(
        all.iter().all(|index| index.uuid != old),
        "old segment gone"
    );
    assert_column_matches_scan(&dataset, "val").await;

    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    let released = dataset
        .frag_reuse_index()
        .await
        .unwrap()
        .is_none_or(|history| history.details.versions.is_empty());
    assert!(released, "the trim releases the reuse version");
    assert_column_matches_scan(&dataset, "val").await;
}

/// Restore the table to `version` and return the restored handle.
async fn restore(dataset: &Dataset, version: u64) -> Dataset {
    let mut old = dataset.checkout_version(version).await.unwrap();
    old.restore().await.unwrap();
    old
}

async fn cleanup_all_but_latest(dataset: Dataset, uri: &str) -> Dataset {
    let policy = CleanupPolicyBuilder::default()
        .retain_n_versions(&dataset, 1)
        .await
        .unwrap()
        .build();
    dataset.cleanup(policy).execute().await.unwrap();
    let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
    assert_eq!(dataset.versions().await.unwrap().len(), 1);
    dataset
}

/// The row id of `id` (a fragment's rows are its addresses here).
fn row_of(id: i32) -> u64 {
    (((id / ROWS) as u64) << 32) | (id % ROWS) as u64
}

/// 4a. A source built after a deletion summarizes the live rows only; once
///     the table is restored to before the deletion its index list no longer
///     holds that source, and a rebuild sees the restored row.
#[tokio::test]
async fn test_restore_before_deletion_drops_the_later_source() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = write(uri, 2).await;
    let before_deletion = dataset.version().version;
    // Row 5 is a null `val`.
    assert_eq!(val(5), None);
    dataset.delete("id = 5").await.unwrap();
    build(&mut dataset, "val", &default_params()).await;
    let (report, _) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_eq!(
        report.stored_fragments,
        vec![0, 1],
        "aligned bounds cover f0"
    );
    assert!(
        !ids_matching(&dataset, "val IS NULL", true)
            .await
            .contains(&row_of(5))
    );
    assert_column_matches_scan(&dataset, "val").await;

    let mut restored = restore(&dataset, before_deletion).await;
    assert!(
        crate::index::loading::load_all_indices(&restored)
            .await
            .unwrap()
            .is_empty(),
        "the restored manifest lists no index"
    );
    let (report, _) = rebuild(&mut restored, "val", &default_params()).await;
    assert!(report.stored_fragments.is_empty());
    assert_eq!(report.scanned_fragments, vec![0, 1]);
    assert!(
        ids_matching(&restored, "val IS NULL", true)
            .await
            .contains(&row_of(5))
    );
    assert_column_matches_scan(&restored, "val").await;

    // The same after the old versions are cleaned up.
    let restored = restore(&restored, before_deletion).await;
    let mut cleaned = cleanup_all_but_latest(restored, uri).await;
    assert!(
        crate::index::loading::load_all_indices(&cleaned)
            .await
            .unwrap()
            .is_empty()
    );
    let (report, _) = rebuild(&mut cleaned, "val", &default_params()).await;
    assert!(report.stored_fragments.is_empty());
    assert!(
        ids_matching(&cleaned, "val IS NULL", true)
            .await
            .contains(&row_of(5))
    );
    assert_column_matches_scan(&cleaned, "val").await;
}

/// 4b. A source built before a deletion still describes a population that
///     contains every live row; after a restore to before the deletion the
///     restored source is reused and the restored row is found.
#[tokio::test]
async fn test_restore_before_deletion_reuses_the_earlier_source() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = write(uri, 2).await;
    build(&mut dataset, "val", &default_params()).await;
    let indexed = dataset.version().version;
    let source = segments(&dataset, "val").await[0].uuid;
    dataset.delete("id = 5").await.unwrap();
    let (report, _) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_eq!(report.stored_fragments, vec![0, 1]);
    assert_column_matches_scan(&dataset, "val").await;

    let mut restored = restore(&dataset, indexed).await;
    let listed: Vec<Uuid> = segments(&restored, "val")
        .await
        .iter()
        .map(|s| s.uuid)
        .collect();
    assert_eq!(listed, vec![source]);
    let (report, _) = rebuild(&mut restored, "val", &default_params()).await;
    assert_eq!(report.stored_fragments, vec![0, 1]);
    assert!(
        ids_matching(&restored, "val IS NULL", true)
            .await
            .contains(&row_of(5))
    );
    assert_column_matches_scan(&restored, "val").await;

    let restored = restore(&restored, indexed).await;
    let mut cleaned = cleanup_all_but_latest(restored, uri).await;
    let (report, _) = rebuild(&mut cleaned, "val", &default_params()).await;
    assert_eq!(report.stored_fragments, vec![0, 1]);
    assert!(
        ids_matching(&cleaned, "val IS NULL", true)
            .await
            .contains(&row_of(5))
    );
    assert_column_matches_scan(&cleaned, "val").await;
}

/// 5. Reuse from a reused output, twice, then restore to each intermediate
///    version: results equal a scan each time, and a rebuild there reuses.
#[tokio::test]
async fn test_reuse_chain_survives_restores() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 3).await;
    let ids = fragment_ids(&dataset);
    build(&mut dataset, "x", &default_params()).await;
    let mut versions = vec![dataset.version().version];
    for _ in 0..2 {
        let (report, _) = rebuild(&mut dataset, "x", &default_params()).await;
        assert_eq!(report.stored_fragments, ids);
        versions.push(dataset.version().version);
    }
    assert_column_matches_scan(&dataset, "x").await;
    for version in versions {
        let mut restored = restore(&dataset, version).await;
        assert_column_matches_scan(&restored, "x").await;
        let (report, _) = rebuild(&mut restored, "x", &default_params()).await;
        assert_eq!(report.stored_fragments, ids, "restored to {version}");
        assert_column_matches_scan(&restored, "x").await;
        dataset = restored;
    }
}

/// Every fragment was scanned and none taken from a segment.
fn assert_all_scanned(report: &SeedReuseReport, ids: &[u64]) {
    assert!(report.stored_fragments.is_empty(), "{report:?}");
    assert_eq!(report.scanned_fragments, ids, "{report:?}");
}

/// 6a. Another index on the same column is not a source.
#[tokio::test]
async fn test_other_index_name_is_not_a_source() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    let ids = fragment_ids(&dataset);
    build_named(&mut dataset, "val", "val_first", &default_params()).await;
    io(&dataset);
    let (_, reports) = with_reports(build_named(
        &mut dataset,
        "val",
        "val_second",
        &default_params(),
    ))
    .await;
    let io = io(&dataset);
    assert_all_scanned(&reports[0], &ids);
    assert!(!data_reads(&dataset, &io, &ids).is_empty());
    assert_matches_scan(&dataset, VAL_FILTERS, &[]).await;
}

/// A version 0 file over the `val` values of every fragment, committed in
/// place of the index's version 1 segment.
async fn replace_with_v0_segment(dataset: &mut Dataset) -> Uuid {
    let current = segments(dataset, "val").await;
    let fragments: Vec<u32> = fragment_ids(dataset).iter().map(|id| *id as u32).collect();
    let uuid = Uuid::new_v4();
    let store = LanceIndexStore::from_dataset_for_new(dataset, &uuid).unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(VALUE_COLUMN_NAME, DataType::Int64, true),
        ArrowField::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let batches: Vec<_> = fragments
        .iter()
        .map(|fragment| {
            let start = *fragment as i32 * ROWS;
            Ok(RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from_iter((start..start + ROWS).map(val))),
                    Arc::new(UInt64Array::from_iter_values(
                        (0..ROWS as u64).map(|row| (u64::from(*fragment) << 32) | row),
                    )),
                ],
            )
            .unwrap())
        })
        .collect();
    let mut builder =
        ZoneMapIndexBuilder::try_new(ZoneMapIndexBuilderParams::new(T), DataType::Int64)
            .unwrap()
            .with_version(ZoneMapIndexVersion::V0);
    builder
        .train(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches),
        )))
        .await
        .unwrap();
    let files = builder.write_index(&store).await.unwrap();
    let v0 = IndexMetadata {
        uuid,
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(RoaringBitmap::from_iter(fragments)),
        index_version: ZoneMapIndexVersion::V0 as i32,
        files: Some(index_files_to_table(files)),
        created_at: None,
        base_id: None,
        ..current[0].clone()
    };
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![v0],
            removed_indices: current,
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();
    uuid
}

/// 6b. A version 0 source is never promoted; it is not even opened.
#[tokio::test]
async fn test_version_0_source_is_recomputed() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    let ids = fragment_ids(&dataset);
    build(&mut dataset, "val", &default_params()).await;
    let v0 = replace_with_v0_segment(&mut dataset).await;
    assert_eq!(segments(&dataset, "val").await[0].index_version, 0);
    let (report, io) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_all_scanned(&report, &ids);
    assert_eq!(report.unusable_sources, 1);
    assert_eq!(segment_reads(&io, &v0), 0, "the version 0 file is not read");
    let now = segments(&dataset, "val").await;
    assert_eq!(now[0].index_version, ZoneMapIndexVersion::CURRENT as i32);
    assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;
}

/// Commit, in place of the index's segment, a segment with the same name,
/// field and details, the current version and its file listed at its true
/// size, whose `zonemap.lance` is a valid Lance file without zone columns.
async fn replace_with_undecodable_segment(dataset: &mut Dataset) -> Uuid {
    use lance_index::scalar::IndexStore;

    let current = segments(dataset, "val").await;
    let fragments: Vec<u32> = fragment_ids(dataset).iter().map(|id| *id as u32).collect();
    let uuid = Uuid::new_v4();
    let store = LanceIndexStore::from_dataset_for_new(dataset, &uuid).unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "unrelated",
        DataType::Int64,
        false,
    )]));
    let mut writer = store
        .new_index_file("zonemap.lance", schema.clone())
        .await
        .unwrap();
    writer
        .write_record_batch(
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 3]))]).unwrap(),
        )
        .await
        .unwrap();
    let file = writer.finish().await.unwrap();
    assert!(file.size_bytes > 0);
    let segment = IndexMetadata {
        uuid,
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(RoaringBitmap::from_iter(fragments)),
        index_version: ZoneMapIndexVersion::CURRENT as i32,
        files: Some(index_files_to_table(vec![file])),
        created_at: None,
        base_id: None,
        ..current[0].clone()
    };
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![segment],
            removed_indices: current,
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();
    uuid
}

/// 6b'. An admitted source whose file does not decode as zones is not a
///      source: every fragment is scanned and the segment is replaced.
#[tokio::test]
async fn test_undecodable_source_falls_back_to_scan() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    let ids = fragment_ids(&dataset);
    build(&mut dataset, "val", &default_params()).await;
    let bad = replace_with_undecodable_segment(&mut dataset).await;
    // The file is read without error and rejected, so the build falls back.
    let store =
        LanceIndexStore::from_dataset_for_existing(&dataset, &segments(&dataset, "val").await[0])
            .await
            .unwrap();
    let outcome = lance_index::scalar::zonemap::StoredZones::load(
        &store,
        ZoneMapIndexVersion::CURRENT,
        &RoaringBitmap::from_iter([0u32, 1]),
    )
    .await
    .unwrap();
    assert!(
        matches!(outcome, StoredZonesLoad::Rejected(_)),
        "{outcome:?}"
    );

    let (report, io) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_all_scanned(&report, &ids);
    assert_eq!(report.unusable_sources, 1, "{report:?}");
    assert_eq!(report.stored_sources, 0, "{report:?}");
    assert_eq!(report.stored_bytes, 0, "{report:?}");
    assert_eq!(report.stored_payload_bytes, 0, "{report:?}");
    assert!(segment_reads(&io, &bad) > 0, "the file was opened");
    let now = segments(&dataset, "val").await;
    assert_eq!(now.len(), 1);
    assert_ne!(now[0].uuid, bad, "the segment is replaced");
    assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;
}

/// 6b''. A storage failure while loading an admitted source fails the
///       build; the table keeps the old segment.
#[tokio::test]
async fn test_missing_source_file_fails_the_build() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    build(&mut dataset, "val", &default_params()).await;
    let old = segments(&dataset, "val").await[0].uuid;
    let path = dataset
        .indices_dir()
        .join(old.to_string())
        .join("zonemap.lance");
    dataset.object_store.delete(&path).await.unwrap();

    let mut dataset = DatasetBuilder::from_uri(dir.as_str()).load().await.unwrap();
    let result = CreateIndexBuilder::new(
        &mut dataset,
        &["val"],
        IndexType::ZoneMap,
        &default_params(),
    )
    .name(index_name("val"))
    .replace(true)
    .await;
    result.expect_err("the build fails");
    let dataset = DatasetBuilder::from_uri(dir.as_str()).load().await.unwrap();
    let now: Vec<Uuid> = segments(&dataset, "val")
        .await
        .iter()
        .map(|segment| segment.uuid)
        .collect();
    assert_eq!(now, vec![old]);
}

/// 6c. A different `rows_per_zone` or null-position setting makes the source
///     unusable; every fragment is recomputed.
#[tokio::test]
async fn test_configuration_mismatch_is_recomputed() {
    // (source params, build params, whether the mismatch is visible in the
    // segment metadata so the file is not opened)
    let cases = [
        (params(T, false, true), params(200, false, true), true),
        (params(T, false, false), params(T, false, true), true),
        (params(T, false, true), params(T, false, false), false),
    ];
    for (i, (source, target, from_metadata)) in cases.into_iter().enumerate() {
        let dir = TempStrDir::default();
        let mut dataset = write(dir.as_str(), 2).await;
        let ids = fragment_ids(&dataset);
        build(&mut dataset, "val", &source).await;
        let old = segments(&dataset, "val").await[0].uuid;
        let (report, io) = rebuild(&mut dataset, "val", &target).await;
        assert_all_scanned(&report, &ids);
        assert_eq!(report.unusable_sources, 1, "case {i}");
        assert_eq!(segment_reads(&io, &old) == 0, from_metadata, "case {i}");
        assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;
    }
}

/// 6d. A fragment with an overlay on the indexed field newer than the
///     source is recomputed; an overlay on another column does not matter.
#[tokio::test]
async fn test_newer_overlay_on_the_field_is_recomputed() {
    use super::dataset_overlay_index_masking::commit_overlay;
    use lance_table::format::overlay::OverlayCoverage;

    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 3).await;
    build(&mut dataset, "val", &default_params()).await;
    let val_field = dataset.schema().field("val").unwrap().id;
    let name_field = dataset.schema().field("name").unwrap().id;
    // Fragment 1, offset 0: val becomes 10,000,000, beyond every stored bound.
    let dataset = commit_overlay(
        dataset,
        "val_overlay",
        1,
        &[val_field],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0])),
        vec![Arc::new(Int64Array::from(vec![Some(10_000_000)]))],
    )
    .await;
    let mut dataset = commit_overlay(
        dataset,
        "name_overlay",
        2,
        &[name_field],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0])),
        vec![Arc::new(StringArray::from(vec![Some("overlaid")]))],
    )
    .await;
    let (report, _) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_eq!(report.stored_fragments, vec![0, 2]);
    assert_eq!(report.scanned_fragments, vec![1]);
    assert_eq!(
        ids_matching(&dataset, "val = 10000000", true).await,
        BTreeSet::from([row_of(ROWS)])
    );
    assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;
}

/// 6e. A fragment two committed segments both offer is taken from neither.
#[tokio::test]
async fn test_fragment_offered_by_two_segments_is_recomputed() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 3).await;
    // Two uncommitted builds whose coverage overlaps on fragment 1.
    let mut built = Vec::new();
    for fragments in [vec![0, 1], vec![1, 2]] {
        built.push(
            CreateIndexBuilder::new(
                &mut dataset,
                &["val"],
                IndexType::ZoneMap,
                &default_params(),
            )
            .name(index_name("val"))
            .fragments(fragments)
            .execute_uncommitted()
            .await
            .unwrap(),
        );
    }
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: built.clone(),
            removed_indices: vec![],
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();
    let listed = crate::index::loading::load_all_indices_by_name(&dataset, &index_name("val"))
        .await
        .unwrap();
    assert_eq!(listed.len(), 2, "both overlapping segments are committed");

    let (report, _) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_eq!(report.stored_fragments, vec![0, 2]);
    assert_eq!(report.scanned_fragments, vec![1]);
    assert_eq!(report.stored_sources, 2);
    assert_eq!(segments(&dataset, "val").await.len(), 1);
    assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;
}

fn zone_file_size(segment: &IndexMetadata) -> u64 {
    segment
        .files
        .as_ref()
        .unwrap()
        .iter()
        .find(|file| file.path == "zonemap.lance")
        .unwrap()
        .size_bytes
}

/// 7. A source above the admission limit is skipped without being opened;
///    one exactly at the limit is used.
#[tokio::test]
async fn test_admission_limit_skips_without_opening() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    let ids = fragment_ids(&dataset);
    build(&mut dataset, "val", &default_params()).await;
    let source = segments(&dataset, "val").await[0].clone();
    let size = zone_file_size(&source);

    let (report, io) = TEST_STORED_ZONE_ADMISSION_BYTES
        .scope(size - 1, rebuild(&mut dataset, "val", &default_params()))
        .await;
    assert_all_scanned(&report, &ids);
    assert_eq!(report.skipped_sources, 1);
    assert_eq!(report.stored_bytes, 0);
    assert_eq!(segment_reads(&io, &source.uuid), 0, "not opened");
    assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;

    let size = zone_file_size(&segments(&dataset, "val").await[0]);
    let (report, _) = TEST_STORED_ZONE_ADMISSION_BYTES
        .scope(size, rebuild(&mut dataset, "val", &default_params()))
        .await;
    assert_eq!(report.stored_fragments, ids);
    assert_eq!(report.stored_bytes, size);
    assert_eq!(report.skipped_sources, 0);
    assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;
}

/// 8a. Discovery is disabled on tables with stable row ids.
#[tokio::test]
async fn test_stable_row_ids_disable_discovery() {
    let dir = TempStrDir::default();
    let mut dataset = write_with(dir.as_str(), 2, true).await;
    let ids = fragment_ids(&dataset);
    build(&mut dataset, "val", &default_params()).await;
    let (report, _) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_all_scanned(&report, &ids);
    assert_eq!(report.stored_sources + report.unusable_sources, 0);
    assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;
}

/// 8b. Discovery is disabled when the fragment reuse history is tagged. The
///     manifest of an otherwise identical handle carries the sticky tagged
///     history flag; an uncommitted build on it reuses nothing while the
///     same build on the untagged handle reuses every fragment.
#[tokio::test]
async fn test_tagged_history_disables_discovery() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    let ids = fragment_ids(&dataset);
    build(&mut dataset, "val", &default_params()).await;

    async fn uncommitted(dataset: &mut Dataset) -> SeedReuseReport {
        let (_, reports) = with_reports(async {
            CreateIndexBuilder::new(dataset, &["val"], IndexType::ZoneMap, &default_params())
                .name(index_name("val"))
                .replace(true)
                .execute_uncommitted()
                .await
                .unwrap()
        })
        .await;
        reports.into_iter().next().unwrap()
    }

    let mut tagged = dataset.clone();
    let mut manifest = tagged.manifest.as_ref().clone();
    manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
    tagged.manifest = Arc::new(manifest);
    let report = uncommitted(&mut tagged).await;
    assert_all_scanned(&report, &ids);

    let report = uncommitted(&mut dataset).await;
    assert_eq!(report.stored_fragments, ids);
}

/// 8c. Caller-supplied training rows bypass the seed-aware build, and with
///     it the stored zones.
#[tokio::test]
async fn test_caller_supplied_training_data_disables_discovery() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    build(&mut dataset, "val", &default_params()).await;
    let criteria = TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr();
    let training = crate::index::scalar::load_training_data(
        &dataset,
        "val",
        &criteria,
        None,
        true,
        None,
        Default::default(),
    )
    .await
    .unwrap();
    let (_, reports) = with_reports(async {
        CreateIndexBuilder::new(
            &mut dataset,
            &["val"],
            IndexType::ZoneMap,
            &default_params(),
        )
        .name(index_name("val"))
        .replace(true)
        .preprocessed_stream(training, criteria.clone())
        .await
        .unwrap()
    })
    .await;
    assert!(
        reports.is_empty(),
        "the seed-aware build did not run: {reports:?}"
    );
    assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;
}

/// 6f. A read of the admitted source's zone file that fails after its
///     metadata was opened fails the build, whatever error the reader
///     returns; the table keeps the old segment.
#[tokio::test]
async fn test_failed_read_after_open_fails_the_build() {
    use crate::dataset::ReadParams;
    use crate::utils::test::FailingProxyStore;
    use lance_io::object_store::ObjectStoreParams;

    let dir = TempStrDir::default();
    let prefix = if dir.as_str().starts_with('/') {
        ""
    } else {
        "/"
    };
    // Routed through the object store so that the proxy sees every read.
    let uri = format!("file-object-store://{prefix}{}", dir.as_str());
    let mut dataset = write(&uri, 2).await;
    // Zones of two rows make the zone file larger than the first read of a
    // file (its tail, where the metadata is), so its records are read by a
    // later request.
    let fine = params(2, false, true);
    build(&mut dataset, "val", &fine).await;
    let old = segments(&dataset, "val").await[0].clone();
    assert!(zone_file_size(&old) > 2 * 4096, "{}", zone_file_size(&old));
    let file: &'static str = Box::leak(format!("{}/zonemap.lance", old.uuid).into_boxed_str());

    // A cold handle whose reads of the zone file fail after the first one.
    // The manifest records the file's size, so the first request is the
    // tail read that opens it.
    let failing_handle = || async {
        let failing = Arc::new(FailingProxyStore::new());
        let dataset = DatasetBuilder::from_uri(&uri)
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(failing.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        failing.fail_after_n("get_opts", file, 1, "injected zone file read failure");
        failing.fail_after_n("get_ranges", file, 0, "injected zone file read failure");
        dataset
    };

    // Under that fault the file opens and its records do not read.
    {
        use lance_index::scalar::IndexStore;
        let dataset = failing_handle().await;
        let store = LanceIndexStore::from_dataset_for_existing(&dataset, &old)
            .await
            .unwrap();
        let reader = store.open_index_file("zonemap.lance").await.unwrap();
        let error = reader
            .read_range(0..reader.num_rows(), None)
            .await
            .expect_err("the record read fails");
        assert!(error.to_string().contains("injected"), "{error}");
    }

    let mut dataset = failing_handle().await;
    let (result, reports) = with_reports(async {
        CreateIndexBuilder::new(&mut dataset, &["val"], IndexType::ZoneMap, &fine)
            .name(index_name("val"))
            .replace(true)
            .await
    })
    .await;
    let error = result.expect_err("the build fails");
    assert!(
        error
            .to_string()
            .contains("injected zone file read failure"),
        "{error}"
    );
    assert!(reports.is_empty(), "{reports:?}");

    let dataset = DatasetBuilder::from_uri(&uri).load().await.unwrap();
    let now: Vec<Uuid> = segments(&dataset, "val")
        .await
        .iter()
        .map(|segment| segment.uuid)
        .collect();
    assert_eq!(now, vec![old.uuid]);
}

/// Filters on `x` for the staged-publication tests. A scan selects the NaN
/// and the -NaN rows with `x > 1e308`; `x < -1e308` does not select the
/// -NaN rows.
const X_FILTERS: &[&str] = &[
    "x > 10",
    "x < -40",
    "x = 3",
    "x >= 0 AND x <= 1",
    "x IS NULL",
    "x IS NOT NULL",
    "x > 1e308",
    "x < -1e308",
    "x > 100000000",
];

fn sorted(mut ids: Vec<u64>) -> Vec<u64> {
    ids.sort_unstable();
    ids
}

fn uuids(segments: &[IndexMetadata]) -> BTreeSet<Uuid> {
    segments.iter().map(|segment| segment.uuid).collect()
}

/// An uncommitted build of the index on `column` over `fragments` (every
/// fragment when `None`), with its report and the requests it made.
async fn staged_build(
    dataset: &mut Dataset,
    column: &str,
    params: &ScalarIndexParams,
    fragments: Option<Vec<u32>>,
) -> (IndexMetadata, SeedReuseReport, IoStats) {
    io(dataset);
    let (segment, reports) = with_reports(async {
        let mut builder = CreateIndexBuilder::new(dataset, &[column], IndexType::ZoneMap, params)
            .name(index_name(column))
            .replace(true);
        if let Some(fragments) = fragments {
            builder = builder.fragments(fragments);
        }
        builder.execute_uncommitted().await.unwrap()
    })
    .await;
    let io = io(dataset);
    assert_eq!(reports.len(), 1, "{reports:?}");
    (segment, reports.into_iter().next().unwrap(), io)
}

/// 9. Worker-style staged builds over fragment subsets, merged and
///    committed, on a table that already has several disjoint committed
///    segments of the index.
#[tokio::test]
async fn test_staged_subset_builds_merge_and_commit() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 6).await;
    let name = index_name("x");
    let seeded = params(T, true, true);
    let mut committed = Vec::new();
    for fragments in [vec![0, 1], vec![2, 3], vec![4, 5]] {
        let (segment, report, _) = staged_build(&mut dataset, "x", &seeded, Some(fragments)).await;
        assert!(report.stored_fragments.is_empty(), "{report:?}");
        committed.push(segment);
    }
    dataset
        .commit_existing_index_segments(&name, "x", committed.clone())
        .await
        .unwrap();
    assert_eq!(uuids(&segments(&dataset, "x").await), uuids(&committed));
    // Fragment 6 is written after the index exists and carries a write seed.
    append(&mut dataset, 6 * ROWS, ROWS).await;
    assert_eq!(fragment_ids(&dataset), vec![0, 1, 2, 3, 4, 5, 6]);
    let indexed: Vec<u64> = (0..6).collect();

    // Worker 1: one fragment from each of the first two segments.
    let (first, report, io) = staged_build(&mut dataset, "x", &seeded, Some(vec![0, 2])).await;
    assert_eq!(
        sorted(report.stored_fragments.clone()),
        vec![0, 2],
        "{report:?}"
    );
    assert_eq!(report.stored_sources, 2, "{report:?}");
    assert!(report.reused_fragments.is_empty(), "{report:?}");
    assert!(report.scanned_fragments.is_empty(), "{report:?}");
    assert_eq!(report.scanned_rows, 0, "{report:?}");
    assert_eq!(data_reads(&dataset, &io, &indexed), Vec::<String>::new());

    // Worker 2: the other fragments of those segments and the new one.
    let (second, report, io) = staged_build(&mut dataset, "x", &seeded, Some(vec![1, 3, 6])).await;
    assert_eq!(
        sorted(report.stored_fragments.clone()),
        vec![1, 3],
        "{report:?}"
    );
    assert_eq!(report.stored_sources, 2, "{report:?}");
    assert_eq!(report.reused_fragments, vec![6], "{report:?}");
    assert!(report.scanned_fragments.is_empty(), "{report:?}");
    assert_eq!(report.scanned_rows, 0, "{report:?}");
    assert_eq!(data_reads(&dataset, &io, &indexed), Vec::<String>::new());

    let merged = dataset
        .merge_existing_index_segments(vec![first, second])
        .await
        .unwrap();
    assert_eq!(
        merged.fragment_bitmap.clone().unwrap(),
        RoaringBitmap::from_iter([0u32, 1, 2, 3, 6])
    );
    dataset
        .commit_existing_index_segments(&name, "x", vec![merged.clone()])
        .await
        .unwrap();
    // The two segments the merged output covers are replaced; the third,
    // disjoint from it, stays.
    assert_eq!(
        uuids(&segments(&dataset, "x").await),
        BTreeSet::from([merged.uuid, committed[2].uuid])
    );
    assert_matches_scan(&dataset, X_FILTERS, &["x > 10", "x IS NULL"]).await;
}

/// The fragments each listed segment of the index on `x` covers.
async fn coverage(dataset: &Dataset) -> Vec<(Uuid, Vec<u32>)> {
    segments(dataset, "x")
        .await
        .iter()
        .map(|segment| {
            let fragments = segment
                .fragment_bitmap
                .as_ref()
                .map(|bitmap| bitmap.iter().collect())
                .unwrap_or_default();
            (segment.uuid, fragments)
        })
        .collect()
}

fn is_incompatible(result: &crate::Result<()>) -> bool {
    matches!(result, Err(crate::Error::IncompatibleTransaction { .. }))
}

fn is_retryable(result: &crate::Result<()>) -> bool {
    matches!(result, Err(crate::Error::RetryableCommitConflict { .. }))
}

/// 10a. Restore to an older version between the build and its
///      publication, also with the history cleaned up: whatever is
///      published still finds the restored null and NaN rows.
#[tokio::test]
async fn test_staged_commit_after_restore() {
    for cleanup in [false, true] {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = write(uri, 2).await;
        let before_deletion = dataset.version().version;
        assert_eq!(x(13), None);
        assert!(x(14).unwrap().is_nan());
        dataset.delete("id = 13 OR id = 14").await.unwrap();
        // The committed source and the staged build summarize the live rows.
        build(&mut dataset, "x", &default_params()).await;
        let source = segments(&dataset, "x").await[0].uuid;
        let (staged, report, _) = staged_build(&mut dataset, "x", &default_params(), None).await;
        assert_eq!(report.stored_fragments, vec![0, 1], "{report:?}");

        let other = DatasetBuilder::from_uri(uri).load().await.unwrap();
        let restored = restore(&other, before_deletion).await;
        if cleanup {
            cleanup_all_but_latest(restored, uri).await;
        }
        let result = dataset
            .commit_existing_index_segments(&index_name("x"), "x", vec![staged.clone()])
            .await;
        assert!(is_incompatible(&result), "cleanup={cleanup}: {result:?}");

        let latest = DatasetBuilder::from_uri(uri).load().await.unwrap();
        let listed = uuids(&segments(&latest, "x").await);
        assert!(!listed.contains(&staged.uuid), "{listed:?}");
        assert!(!listed.contains(&source), "{listed:?}");
        assert!(
            ids_matching(&latest, "x > 1e308", false)
                .await
                .contains(&row_of(14)),
            "the filter selects the NaN row"
        );
        assert!(
            ids_matching(&latest, "x IS NULL", true)
                .await
                .contains(&row_of(13))
        );
        assert!(
            ids_matching(&latest, "x > 1e308", true)
                .await
                .contains(&row_of(14))
        );
        assert_matches_scan(&latest, X_FILTERS, &[]).await;
    }
}

/// Ids in the second zone of fragment 0 whose `x` is null, NaN or -NaN.
fn zone_null_and_nan_ids() -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let zone = T as i32..2 * T as i32;
    let nulls = zone.clone().filter(|i| x(*i).is_none()).collect();
    let nans = zone
        .clone()
        .filter(|i| x(*i).is_some_and(|v| v.is_nan() && v.is_sign_positive()))
        .collect();
    let negative_nans = zone
        .filter(|i| x(*i).is_some_and(|v| v.is_nan() && v.is_sign_negative()))
        .collect();
    (nulls, nans, negative_nans)
}

/// 10a'. Restore between the build and its publication, then the retrying
///       caller refreshes the handle and commits the same staged output
///       again. Every null and NaN row of one whole zone was deleted before
///       the build, so the staged summary of that zone has none; the
///       restore brings them back. Run for a staged output taken from
///       stored records of a committed source and for one that is not, and
///       with and without the history cleaned up. The second commit
///       succeeds with pruned coverage and every query equals a scan.
async fn staged_commit_again_after_restore(from_stored: bool) {
    let (nulls, nans, negative_nans) = zone_null_and_nan_ids();
    assert!(!nulls.is_empty() && !nans.is_empty() && !negative_nans.is_empty());
    let deleted: Vec<String> = nulls
        .iter()
        .chain(&nans)
        .chain(&negative_nans)
        .map(|id| id.to_string())
        .collect();
    for cleanup in [false, true] {
        let case = format!("from_stored={from_stored} cleanup={cleanup}");
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = write(uri, 2).await;
        let before_deletion = dataset.version().version;
        dataset
            .delete(&format!(
                "id >= {} AND id < {} AND id IN ({})",
                T,
                2 * T,
                deleted.join(", ")
            ))
            .await
            .unwrap();
        if from_stored {
            build(&mut dataset, "x", &default_params()).await;
        }
        let (staged, report, _) = staged_build(&mut dataset, "x", &default_params(), None).await;
        if from_stored {
            assert_eq!(report.stored_fragments, vec![0, 1], "{case}: {report:?}");
        } else {
            assert!(report.stored_fragments.is_empty(), "{case}: {report:?}");
        }

        // The staged record of that zone excludes the deleted rows.
        let store = LanceIndexStore::from_dataset_for_existing(&dataset, &staged)
            .await
            .unwrap();
        let version = ZoneMapIndexVersion::from_metadata(staged.index_version as u32).unwrap();
        let records = ZoneMapIndex::load_zone_statistics(&store, version)
            .await
            .unwrap();
        let column = |name: &str| {
            records[name]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
        };
        let counts = |name: &str| {
            records[name]
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .unwrap()
        };
        let zone: Vec<usize> = (0..records.num_rows())
            .filter(|row| {
                column("fragment_id").value(*row) == 0
                    && (T..2 * T).contains(&column("zone_start").value(*row))
            })
            .collect();
        assert_eq!(zone.len(), 1, "{case}: {records:?}");
        assert_eq!(counts("null_count").value(zone[0]), 0, "{case}");
        assert_eq!(counts("nan_count").value(zone[0]), 0, "{case}");

        let other = DatasetBuilder::from_uri(uri).load().await.unwrap();
        let restored = restore(&other, before_deletion).await;
        if cleanup {
            cleanup_all_but_latest(restored, uri).await;
        }
        let first = dataset
            .commit_existing_index_segments(&index_name("x"), "x", vec![staged.clone()])
            .await;
        assert!(is_incompatible(&first), "{case}: {first:?}");

        dataset.checkout_latest().await.unwrap();
        dataset
            .commit_existing_index_segments(&index_name("x"), "x", vec![staged.clone()])
            .await
            .unwrap();
        let latest = DatasetBuilder::from_uri(uri).load().await.unwrap();
        // The commit prunes fragment 0, whose restored rows the staged
        // summary does not hold; with the history cleaned up it covers none.
        let expected: Vec<u32> = if cleanup { vec![] } else { vec![1] };
        assert_eq!(
            coverage(&latest).await,
            vec![(staged.uuid, expected)],
            "{case}"
        );

        // The restored rows are live and found through the index.
        let null_rows: BTreeSet<u64> = nulls.iter().map(|id| row_of(*id)).collect();
        let nan_rows: BTreeSet<u64> = nans
            .iter()
            .chain(&negative_nans)
            .map(|id| row_of(*id))
            .collect();
        assert!(
            null_rows.is_subset(&ids_matching(&latest, "x IS NULL", false).await),
            "{case}"
        );
        assert!(
            null_rows.is_subset(&ids_matching(&latest, "x IS NULL", true).await),
            "{case}"
        );
        // NaN of either sign is selected by one of the two NaN filters.
        for use_index in [false, true] {
            let mut found = ids_matching(&latest, "x > 1e308", use_index).await;
            found.extend(ids_matching(&latest, "x < -1e308", use_index).await);
            assert!(nan_rows.is_subset(&found), "{case} use_index={use_index}");
        }
        assert_matches_scan(&latest, X_FILTERS, &[]).await;
    }
}

#[tokio::test]
async fn test_staged_commit_again_after_restore_from_stored_records() {
    staged_commit_again_after_restore(true).await;
}

#[tokio::test]
async fn test_staged_commit_again_after_restore_without_stored_records() {
    staged_commit_again_after_restore(false).await;
}

/// 10b. Deferred and eager compaction between the build and its
///      publication; on a failure the same staged output is committed again
///      after refreshing the handle (never rebuilt).
#[tokio::test]
async fn test_staged_commit_after_compaction() {
    for defer in [true, false] {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = write(uri, 2).await;
        append(&mut dataset, 2 * ROWS, ROWS / 4).await;
        append(&mut dataset, 2 * ROWS + ROWS / 4, ROWS / 4).await;
        build(&mut dataset, "x", &default_params()).await;
        let (staged, report, _) = staged_build(&mut dataset, "x", &default_params(), None).await;
        assert_eq!(report.stored_fragments, vec![0, 1, 2, 3], "{report:?}");

        let mut other = DatasetBuilder::from_uri(uri).load().await.unwrap();
        compact_files(
            &mut other,
            CompactionOptions {
                target_rows_per_fragment: ROWS as usize,
                defer_index_remap: defer,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        // Fragments 2 and 3 are joined into fragment 4.
        assert_eq!(fragment_ids(&other), vec![0, 1, 4]);
        let result = dataset
            .commit_existing_index_segments(&index_name("x"), "x", vec![staged.clone()])
            .await;
        let latest = DatasetBuilder::from_uri(uri).load().await.unwrap();
        if defer {
            result.unwrap();
            assert_eq!(coverage(&latest).await, vec![(staged.uuid, vec![0, 1, 4])]);
        } else {
            assert!(is_retryable(&result), "{result:?}");
            let listed = uuids(&segments(&latest, "x").await);
            assert!(
                !listed.contains(&staged.uuid),
                "nothing published: {listed:?}"
            );
            dataset.checkout_latest().await.unwrap();
            dataset
                .commit_existing_index_segments(&index_name("x"), "x", vec![staged.clone()])
                .await
                .unwrap();
            let latest = DatasetBuilder::from_uri(uri).load().await.unwrap();
            // The compaction destination is left unindexed and is scanned.
            assert_eq!(coverage(&latest).await, vec![(staged.uuid, vec![0, 1])]);
        }
        let latest = DatasetBuilder::from_uri(uri).load().await.unwrap();
        assert_matches_scan(&latest, X_FILTERS, &[]).await;
    }
}

/// 10c. Another writer replaces the source segment with the same data
///      under a new uuid: that alone does not invalidate the staged output.
#[tokio::test]
async fn test_staged_commit_after_source_replaced() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = write(uri, 3).await;
    build(&mut dataset, "x", &default_params()).await;
    let (staged, report, _) = staged_build(&mut dataset, "x", &default_params(), None).await;
    assert_eq!(report.stored_fragments, vec![0, 1, 2], "{report:?}");

    let mut other = DatasetBuilder::from_uri(uri).load().await.unwrap();
    build(&mut other, "x", &default_params()).await;
    let result = dataset
        .commit_existing_index_segments(&index_name("x"), "x", vec![staged.clone()])
        .await;
    assert!(is_retryable(&result), "{result:?}");
    dataset.checkout_latest().await.unwrap();
    dataset
        .commit_existing_index_segments(&index_name("x"), "x", vec![staged.clone()])
        .await
        .unwrap();
    let latest = DatasetBuilder::from_uri(uri).load().await.unwrap();
    assert_eq!(
        uuids(&segments(&latest, "x").await),
        BTreeSet::from([staged.uuid])
    );
    assert_matches_scan(&latest, X_FILTERS, &["x > 10", "x IS NULL"]).await;
}

/// 10d. Another writer changes values of the indexed field in a fragment
///      the staged output took from stored records, by rewriting the rows
///      or by an overlay on the field: the published index must not answer
///      for those rows with the old statistics.
#[tokio::test]
async fn test_staged_commit_after_indexed_values_change() {
    use super::dataset_overlay_index_masking::commit_overlay;
    use crate::dataset::UpdateBuilder;
    use lance_table::format::overlay::OverlayCoverage;

    for overlay in [false, true] {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = write(uri, 3).await;
        build(&mut dataset, "x", &default_params()).await;
        let (staged, report, _) = staged_build(&mut dataset, "x", &default_params(), None).await;
        assert_eq!(report.stored_fragments, vec![0, 1, 2], "{report:?}");

        let other = DatasetBuilder::from_uri(uri).load().await.unwrap();
        if overlay {
            let x_field = other.schema().field("x").unwrap().id;
            // Fragment 1, offset 0: x becomes 1e9, beyond every stored bound.
            commit_overlay(
                other,
                "x_overlay",
                1,
                &[x_field],
                OverlayCoverage::dense(RoaringBitmap::from_iter([0])),
                vec![Arc::new(Float64Array::from(vec![Some(1e9)]))],
            )
            .await;
        } else {
            UpdateBuilder::new(Arc::new(other))
                .update_where("id >= 1000 AND id < 1010")
                .unwrap()
                .set("x", "1000000000.0")
                .unwrap()
                .build()
                .unwrap()
                .execute()
                .await
                .unwrap();
        }
        dataset
            .commit_existing_index_segments(&index_name("x"), "x", vec![staged.clone()])
            .await
            .unwrap();

        let latest = DatasetBuilder::from_uri(uri).load().await.unwrap();
        if overlay {
            // The overlay changes fragment 1 in place; it stays covered.
            assert_eq!(fragment_ids(&latest), vec![0, 1, 2]);
            assert_eq!(coverage(&latest).await, vec![(staged.uuid, vec![0, 1, 2])]);
        } else {
            // The rewritten rows move to fragment 3, which is not covered.
            assert_eq!(fragment_ids(&latest), vec![0, 1, 2, 3]);
            assert_eq!(coverage(&latest).await, vec![(staged.uuid, vec![0, 1, 2])]);
        }
        let changed = ids_matching(&latest, "x > 100000000", false).await;
        assert!(!changed.is_empty(), "the changed rows exist");
        assert_eq!(ids_matching(&latest, "x > 100000000", true).await, changed);
        assert_matches_scan(&latest, X_FILTERS, &[]).await;
    }
}

/// 10e. The build's snapshot is cleaned up before the staged commit:
///      the commit succeeds and covers only the fragments it was built on.
#[tokio::test]
async fn test_staged_commit_after_build_snapshot_is_cleaned_up() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = write(uri, 2).await;
    build(&mut dataset, "x", &default_params()).await;
    let (staged, report, _) = staged_build(&mut dataset, "x", &default_params(), None).await;
    assert_eq!(report.stored_fragments, vec![0, 1], "{report:?}");

    let mut other = DatasetBuilder::from_uri(uri).load().await.unwrap();
    append(&mut other, 2 * ROWS, ROWS).await;
    cleanup_all_but_latest(other, uri).await;
    dataset
        .commit_existing_index_segments(&index_name("x"), "x", vec![staged.clone()])
        .await
        .unwrap();

    let latest = DatasetBuilder::from_uri(uri).load().await.unwrap();
    // The appended fragment 2 is not covered.
    assert_eq!(fragment_ids(&latest), vec![0, 1, 2]);
    assert_eq!(coverage(&latest).await, vec![(staged.uuid, vec![0, 1])]);
    assert_matches_scan(&latest, X_FILTERS, &[]).await;
}

/// 11. A committed source made by the merge of segments, whose stored
///     records for a compaction destination are not that fragment's target
///     zones (three records inside one target interval), is discovered and
///     offers the destination; the shape check rejects it and the fragment
///     is recomputed.
#[tokio::test]
async fn test_merge_produced_source_is_rejected_by_shape() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 2).await;
    // Three fragments of 100 rows that compaction joins into one of 300.
    for k in 0..3 {
        append(&mut dataset, 2 * ROWS + k * 100, 100).await;
    }
    build(&mut dataset, "val", &default_params()).await;
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: ROWS as usize,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let ids = fragment_ids(&dataset);
    assert_eq!(ids.len(), 3, "{ids:?}");
    assert_eq!(&ids[..2], &[0, 1]);
    let destination = ids[2];

    let merged = dataset
        .merge_existing_index_segments(segments(&dataset, "val").await)
        .await
        .unwrap();
    assert_eq!(
        merged.fragment_bitmap.clone().unwrap(),
        RoaringBitmap::from_iter(ids.iter().map(|id| *id as u32))
    );
    // Four target zones for each full fragment, and three records of 100
    // rows for the destination, whose single target interval is [0, 300).
    let store = LanceIndexStore::from_dataset_for_existing(&dataset, &merged)
        .await
        .unwrap();
    let records = ZoneMapIndex::load_zone_statistics(&store, ZoneMapIndexVersion::CURRENT)
        .await
        .unwrap();
    assert_eq!(records.num_rows(), 4 + 4 + 3);
    dataset
        .commit_existing_index_segments(&index_name("val"), "val", vec![merged.clone()])
        .await
        .unwrap();
    assert_eq!(
        uuids(&segments(&dataset, "val").await),
        BTreeSet::from([merged.uuid])
    );

    // Only the destination requested: the merged segment offers it (so it
    // is loaded and counted), and nothing is taken from it.
    let (_, report, io) = staged_build(
        &mut dataset,
        "val",
        &default_params(),
        Some(vec![destination as u32]),
    )
    .await;
    assert_eq!(report.stored_sources, 1, "{report:?}");
    assert_eq!(report.unusable_sources, 0, "{report:?}");
    assert!(report.stored_fragments.is_empty(), "{report:?}");
    assert_eq!(report.scanned_fragments, vec![destination], "{report:?}");
    assert!(segment_reads(&io, &merged.uuid) > 0, "the source was read");

    let (report, io) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_eq!(report.stored_fragments, vec![0, 1], "{report:?}");
    assert_eq!(report.stored_sources, 1, "{report:?}");
    assert_eq!(report.scanned_fragments, vec![destination], "{report:?}");
    assert_eq!(data_reads(&dataset, &io, &[0, 1]), Vec::<String>::new());
    assert_matches_scan(&dataset, VAL_FILTERS, &["val > 500"]).await;
}
