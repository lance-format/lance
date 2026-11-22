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
    ZoneMapIndexBuilder, ZoneMapIndexBuilderParams, ZoneMapIndexVersion,
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
        assert!(report.stored_bytes > 0 && report.stored_decoded_bytes > 0);
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
    // The parse error is not a storage error, so the build falls back.
    let store =
        LanceIndexStore::from_dataset_for_existing(&dataset, &segments(&dataset, "val").await[0])
            .await
            .unwrap();
    let error = lance_index::scalar::zonemap::StoredZones::load(
        &store,
        ZoneMapIndexVersion::CURRENT,
        &RoaringBitmap::from_iter([0u32, 1]),
    )
    .await
    .expect_err("the file does not parse as zones");
    assert!(
        matches!(error, lance_core::Error::InvalidInput { .. }),
        "{error:?}"
    );

    let (report, io) = rebuild(&mut dataset, "val", &default_params()).await;
    assert_all_scanned(&report, &ids);
    assert_eq!(report.unusable_sources, 1, "{report:?}");
    assert_eq!(report.stored_sources, 0, "{report:?}");
    assert_eq!(report.stored_bytes, 0, "{report:?}");
    assert_eq!(report.stored_decoded_bytes, 0, "{report:?}");
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
    let error = result.expect_err("the build fails");
    assert!(
        matches!(
            error,
            lance_core::Error::IO { .. } | lance_core::Error::NotFound { .. }
        ),
        "{error:?}"
    );
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
