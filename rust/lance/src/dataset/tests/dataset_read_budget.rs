// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reads under a session read memory limit return what unlimited reads return,
//! for every file version, column shape, page size, limit and batch size, alone
//! and concurrently, and release everything they reserve, including when they
//! are cancelled or fail.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
use arrow_array::RecordBatch;
use arrow_array::types::{Float32Type, Int32Type};
use arrow_schema::{DataType, Field, Fields};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{ArrayGenerator, ArrayGeneratorExt, ByteCount, array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_file::writer::FileWriterOptions;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use lance_io::object_store::ObjectStoreParams;
use lance_io::read_budget::QueryId;
use rstest::rstest;

use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::fragment::FragReadConfig;
use crate::dataset::{ProjectionRequest, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;
use crate::io::exec::filtered_read::{
    FilteredReadExec, FilteredReadOptions, FilteredReadThreadingMode,
};
use crate::io::exec::{LanceScanConfig, LanceScanExec, TakeExec};
use crate::session::Session;
use crate::utils::test::{DatagenExt, FailingProxyStore, FragmentCount, FragmentRowCount};

/// The shapes of column the read budget is checked against.
#[derive(Clone, Copy, Debug)]
enum Shape {
    Int,
    Utf8,
    VarBinary,
    FixedBinary,
    ListOfBinary,
    Vector,
    Struct,
    LargeBinary,
    /// Few distinct strings, dictionary encoded.
    LowCardinality,
    AllNull,
}

impl Shape {
    /// The column's generator and the rows of a table of a few MB.
    fn generator(self) -> (Box<dyn ArrayGenerator>, u64) {
        match self {
            Self::Int => (array::step::<Int32Type>(), 64 * 1024),
            Self::Utf8 => (array::rand_utf8(ByteCount::from(100), false), 16 * 1024),
            Self::VarBinary => (
                array::rand_varbin(ByteCount::from(1), ByteCount::from(8192)),
                2048,
            ),
            Self::FixedBinary => (array::rand_fixedbin(ByteCount::from(2048), true), 4096),
            Self::ListOfBinary => (
                array::rand_list_any(array::rand_fixedbin(ByteCount::from(4096), true), false),
                1024,
            ),
            Self::Vector => (array::rand_vec::<Float32Type>(128.into()), 8192),
            Self::Struct => (
                array::rand_struct(Fields::from(vec![
                    Field::new("a", DataType::Int32, true),
                    Field::new("b", DataType::Utf8, true),
                ])),
                16 * 1024,
            ),
            Self::LowCardinality => (
                array::cycle_utf8_literals(&["red", "green", "blue", "yellow"]),
                64 * 1024,
            ),
            Self::AllNull => (
                array::rand_utf8(ByteCount::from(100), false).with_random_nulls(1.0),
                16 * 1024,
            ),
            // Rows larger than the tighter limits below.
            Self::LargeBinary => (
                array::rand_varbin(ByteCount::from(256 * 1024), ByteCount::from(1024 * 1024)),
                64,
            ),
        }
    }
}

/// Writes `id` and a `value` column of `shape` in `fragments` fragments.
async fn write(
    dir: &str,
    version: LanceFileVersion,
    shape: Shape,
    fragments: u32,
    max_page_bytes: Option<u64>,
) -> u64 {
    let (generator, rows) = shape.generator();
    let rows_per_fragment = (rows / fragments as u64).max(1) as u32;
    let params = WriteParams {
        max_rows_per_file: rows_per_fragment as usize,
        data_storage_version: Some(version),
        file_writer_options: max_page_bytes.map(|max_page_bytes| FileWriterOptions {
            max_page_bytes: Some(max_page_bytes),
            data_cache_bytes: Some(max_page_bytes),
            ..Default::default()
        }),
        ..Default::default()
    };
    gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", generator)
        .into_dataset_with_params(
            dir,
            FragmentCount::from(fragments),
            FragmentRowCount::from(rows_per_fragment),
            Some(params),
        )
        .await
        .unwrap();
    rows_per_fragment as u64 * fragments as u64
}

pub(super) async fn open(dir: &str, limit: Option<u64>) -> Arc<Dataset> {
    let mut session = Session::default();
    if let Some(limit) = limit {
        session = session.with_read_memory_limit(limit);
    }
    // Through the object store, as remote storage is read, rather than the
    // local file reader.
    let dataset = DatasetBuilder::from_uri(format!("file-object-store://{dir}"))
        .with_session(Arc::new(session))
        .load()
        .await
        .unwrap();
    Arc::new(dataset)
}

/// The rows of `batches` in `id` order, so reads that return rows in any
/// order or batching compare equal.
fn by_id(batches: Vec<RecordBatch>) -> Option<RecordBatch> {
    let schema = batches.first()?.schema();
    let batch = concat_batches(&schema, &batches).unwrap();
    let order = sort_to_indices(&batch["id"], None, None).unwrap();
    Some(take_record_batch(&batch, &order).unwrap())
}

fn context() -> Arc<TaskContext> {
    Arc::new(TaskContext::default())
}

/// A whole-fragment scan of `columns` (all when `None`) through `LanceScanExec`.
fn fragment_scan(
    dataset: &Arc<Dataset>,
    columns: Option<&[&str]>,
    config: LanceScanConfig,
) -> LanceScanExec {
    let schema = match columns {
        Some(columns) => dataset.schema().project(columns).unwrap(),
        None => dataset.schema().clone(),
    };
    LanceScanExec::new(
        dataset.clone(),
        dataset.fragments().clone(),
        None,
        Arc::new(schema),
        config,
    )
}

async fn collect(plan: &dyn ExecutionPlan) -> Vec<RecordBatch> {
    plan.execute(0, context())
        .unwrap()
        .try_collect()
        .await
        .unwrap()
}

/// Fails the test if `read` does not finish in `secs` seconds.
async fn within<T>(secs: u64, read: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(secs), read)
        .await
        .expect("a budgeted read hung")
}

/// The read paths a budget governs.
#[derive(Clone, Copy, Debug)]
enum Read {
    /// A full scan.
    Scan,
    /// A filtered scan that loads `value` only for matching rows.
    FilteredScan,
    /// A scan with a limit and an offset.
    LimitedScan,
    /// `Dataset::take` of scattered rows, with duplicates and out of order.
    Take,
    /// A whole-fragment scan through `LanceScanExec`.
    FragmentScan,
    /// Row addresses from a scan, then `value` through `TakeExec`.
    TakeExec,
    /// A filtered read split into partitions read at once.
    Partitions,
}

const READS: [Read; 7] = [
    Read::Scan,
    Read::FilteredScan,
    Read::LimitedScan,
    Read::Take,
    Read::FragmentScan,
    Read::TakeExec,
    Read::Partitions,
];

async fn read(dataset: &Arc<Dataset>, read: Read, batch_size: usize, rows: u64) -> RecordBatch {
    let batches: Vec<RecordBatch> = match read {
        Read::Scan | Read::FilteredScan | Read::LimitedScan => {
            let mut scanner = dataset.scan();
            scanner
                .batch_size(batch_size)
                .project(&["id", "value"])
                .unwrap();
            match read {
                Read::FilteredScan => {
                    scanner.filter("id % 7 = 3").unwrap();
                }
                Read::LimitedScan => {
                    scanner.limit(Some(10), Some(5)).unwrap();
                }
                _ => {}
            }
            vec![scanner.try_into_batch().await.unwrap()]
        }
        Read::Take => {
            // Rows that exist after any deletions.
            let rows = rows.min(dataset.count_rows(None).await.unwrap() as u64);
            let mut rows: Vec<u64> = (0..rows).step_by(13).collect();
            rows.reverse();
            rows.extend_from_slice(&[0, 0, rows[0]]);
            let projection = ProjectionRequest::from_columns(["id", "value"], dataset.schema());
            // In request order: compared as is, not sorted.
            return dataset.take(&rows, projection).await.unwrap();
        }
        Read::FragmentScan => {
            let config = LanceScanConfig {
                batch_size,
                ..Default::default()
            };
            collect(&fragment_scan(dataset, Some(&["id", "value"]), config)).await
        }
        Read::TakeExec => {
            let config = LanceScanConfig {
                batch_size,
                with_row_address: true,
                ..Default::default()
            };
            let scan = fragment_scan(dataset, Some(&["id"]), config);
            let value = dataset
                .empty_projection()
                .union_column("value", lance_core::datatypes::OnMissing::Error)
                .unwrap();
            let take = TakeExec::try_new(dataset.clone(), Arc::new(scan), value)
                .unwrap()
                .unwrap();
            collect(&take)
                .await
                .into_iter()
                .map(|batch| {
                    let schema = batch.schema();
                    let columns = [
                        schema.index_of("id").unwrap(),
                        schema.index_of("value").unwrap(),
                    ];
                    batch.project(&columns).unwrap()
                })
                .collect()
        }
        Read::Partitions => {
            let mut options = FilteredReadOptions::basic_full_read(dataset)
                .with_batch_size(batch_size as u32)
                .with_projection(
                    dataset
                        .empty_projection()
                        .union_columns(["id", "value"], lance_core::datatypes::OnMissing::Error)
                        .unwrap(),
                );
            options.threading_mode = FilteredReadThreadingMode::MultiplePartitions(3);
            let plan = FilteredReadExec::try_new(dataset.clone(), options, None).unwrap();
            let context = context();
            let partitions = futures::future::try_join_all((0..3).map(|partition| {
                plan.execute(partition, context.clone())
                    .unwrap()
                    .try_collect::<Vec<_>>()
            }))
            .await
            .unwrap();
            partitions.into_iter().flatten().collect()
        }
    };
    by_id(batches).unwrap_or_else(|| RecordBatch::new_empty(Arc::new(dataset.schema().into())))
}

async fn read_all(dataset: &Arc<Dataset>, batch_size: usize, rows: u64) -> Vec<RecordBatch> {
    let mut results = Vec::new();
    for kind in READS {
        results.push(read(dataset, kind, batch_size, rows).await);
    }
    results
}

pub(super) fn assert_released(dataset: &Dataset) {
    let budget = dataset.session.read_budget().unwrap();
    assert_eq!(budget.reserved_bytes(), 0, "a read kept a reservation");
}

/// Waits for spawned reads of dropped streams to wind down and release.
pub(super) async fn assert_released_eventually(dataset: &Dataset) {
    let budget = dataset.session.read_budget().unwrap();
    for _ in 0..1000 {
        if budget.reserved_bytes() == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{} bytes stayed reserved", budget.reserved_bytes());
}

/// Every read path, alone and all at once, returns exactly what it returns
/// without a limit, for every file version and column shape: under a limit
/// every unit overdraws (1 byte), one that rarely binds, and one that never
/// does, which must make no extra requests.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_budgeted_reads_match_unbudgeted(
    #[values(1, 64 * 1024 * 1024, u64::MAX)] limit: u64,
    #[values(
        LanceFileVersion::V2_0,
        LanceFileVersion::V2_1,
        LanceFileVersion::V2_2,
        LanceFileVersion::V2_3
    )]
    version: LanceFileVersion,
    #[values(
        Shape::Int,
        Shape::Utf8,
        Shape::VarBinary,
        Shape::FixedBinary,
        Shape::ListOfBinary,
        Shape::Vector,
        Shape::Struct,
        Shape::LargeBinary,
        Shape::LowCardinality,
        Shape::AllNull
    )]
    shape: Shape,
) {
    let dir = TempStrDir::default();
    let rows = write(&dir, version, shape, 4, None).await;
    let unbudgeted = open(&dir, None).await;
    unbudgeted.object_store.io_stats_incremental();
    let expected = read_all(&unbudgeted, 1024, rows).await;
    let requests = unbudgeted.object_store.io_stats_incremental().read_iops;
    let dataset = open(&dir, Some(limit)).await;
    let budget = dataset.session.read_budget().unwrap();
    within(300, async {
        if limit == u64::MAX {
            // Read first, so caches are as cold as for the unbudgeted read.
            dataset.object_store.io_stats_incremental();
            assert_eq!(read_all(&dataset, 1024, rows).await, expected);
            assert_eq!(
                dataset.object_store.io_stats_incremental().read_iops,
                requests,
                "a budget that never binds added requests"
            );
        }
        budget.reset_peak();
        let concurrent = futures::future::join_all(READS.iter().map(|kind| {
            let dataset = dataset.clone();
            async move { read(&dataset, *kind, 1024, rows).await }
        }))
        .await;
        assert_eq!(concurrent, expected);
    })
    .await;
    // Where chunks are small next to the limit, reads stay close to it.
    let peak = budget.stats().peak_reserved_bytes;
    if limit == 64 * 1024 * 1024 {
        assert!(
            peak as f64 <= 1.5 * limit as f64,
            "reads reserved {peak} bytes under a limit of {limit}"
        );
    }
    assert_released(&dataset);
}

/// Every read path returns what it returns without a limit for any batch size,
/// whether every unit overdraws, units wait for each other, or they fit.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_budgeted_reads_of_any_batch_size(#[values(3, 1024, 65536)] batch_size: usize) {
    let dir = TempStrDir::default();
    let rows = write(&dir, LanceFileVersion::V2_2, Shape::VarBinary, 4, None).await;
    let expected = read_all(&open(&dir, None).await, batch_size, rows).await;
    for limit in [1, 4 * 1024 * 1024, 64 * 1024 * 1024] {
        let dataset = open(&dir, Some(limit)).await;
        let actual = within(300, read_all(&dataset, batch_size, rows)).await;
        assert_eq!(actual, expected, "limit {limit}");
        assert_released(&dataset);
    }
}

/// Many queries at once, each a different read path, under a limit far below
/// what they want together.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_many_concurrent_budgeted_reads() {
    let dir = TempStrDir::default();
    let rows = write(&dir, LanceFileVersion::V2_2, Shape::LargeBinary, 8, None).await;
    let expected = read_all(&open(&dir, None).await, 1024, rows).await;
    let dataset = open(&dir, Some(4 * 1024 * 1024)).await;
    let reads = (0..4 * READS.len()).map(|query| {
        let (dataset, kind) = (dataset.clone(), READS[query % READS.len()]);
        async move { (query % READS.len(), read(&dataset, kind, 1024, rows).await) }
    });
    let results = within(300, futures::future::join_all(reads)).await;
    for (kind, result) in results {
        assert_eq!(result, expected[kind], "{:?}", READS[kind]);
    }
    assert_released(&dataset);
}

/// Tiny pages, pages larger than the limit, and hundreds of tiny fragments.
#[rstest]
#[case::tiny_pages(Shape::Utf8, 4, Some(4096))]
#[case::huge_pages(Shape::FixedBinary, 4, Some(1 << 30))]
#[case::many_fragments(Shape::FixedBinary, 500, None)]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_budgeted_reads_of_extreme_layouts(
    #[case] shape: Shape,
    #[case] fragments: u32,
    #[case] max_page_bytes: Option<u64>,
) {
    let dir = TempStrDir::default();
    let rows = write(
        &dir,
        LanceFileVersion::V2_2,
        shape,
        fragments,
        max_page_bytes,
    )
    .await;
    let expected = read_all(&open(&dir, None).await, 1024, rows).await;
    let dataset = open(&dir, Some(4 * 1024 * 1024)).await;
    assert_eq!(within(300, read_all(&dataset, 1024, rows)).await, expected);
    assert_released(&dataset);
}

/// Writes vectors with a BTree index on `id` and an IVF_FLAT index on `value`.
async fn write_indexed(dir: &str) -> u64 {
    let rows = write(dir, LanceFileVersion::V2_2, Shape::Vector, 4, None).await;
    let mut dataset = (*open(dir, None).await).clone();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["value"],
            IndexType::Vector,
            None,
            &VectorIndexParams::ivf_flat(4, lance_linalg::distance::MetricType::L2),
            true,
        )
        .await
        .unwrap();
    rows
}

fn query_vector() -> arrow_array::Float32Array {
    arrow_array::Float32Array::from(vec![0.5; 128])
}

/// Index-driven reads: a scalar-index filter loads scattered rows, and a
/// vector search refines its candidates with a take.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_budgeted_index_reads() {
    let dir = TempStrDir::default();
    let rows = write_indexed(&dir).await;
    let query = query_vector();
    let search = |dataset: Arc<Dataset>| {
        let query = query.clone();
        async move {
            let mut filtered = dataset.scan();
            filtered
                .filter("id < 50 OR id = 4000 OR id > 8100")
                .unwrap()
                .project(&["id", "value"])
                .unwrap();
            let filtered = by_id(vec![filtered.try_into_batch().await.unwrap()]);
            let mut knn = dataset.scan();
            knn.nearest("value", &query, 10)
                .unwrap()
                .refine(4)
                .project(&["id"])
                .unwrap();
            let knn = by_id(vec![knn.try_into_batch().await.unwrap()]);
            (filtered, knn)
        }
    };
    let expected = search(open(&dir, None).await).await;
    assert_eq!(
        expected.0.as_ref().map(RecordBatch::num_rows),
        Some(50 + 1 + (rows as usize - 8101))
    );
    for limit in [1, 4 * 1024 * 1024] {
        let dataset = open(&dir, Some(limit)).await;
        let actual = within(300, search(dataset.clone())).await;
        assert_eq!(actual, expected, "limit {limit}");
        assert_released(&dataset);
    }
}

/// Index searches that read no table data, so only their index loads reserve.
#[derive(Clone, Copy, Debug)]
enum IndexSearch {
    /// An IVF search, which loads partitions.
    Vector,
    /// A BTree equality filter, which loads a page.
    BTree,
}

/// The row ids `search` finds, without reading any column.
async fn index_search(dataset: &Arc<Dataset>, search: IndexSearch) -> RecordBatch {
    let mut scanner = dataset.scan();
    match search {
        IndexSearch::Vector => {
            scanner.nearest("value", &query_vector(), 10).unwrap();
        }
        IndexSearch::BTree => {
            scanner.filter("id = 4000").unwrap();
        }
    }
    scanner.with_row_id().project::<&str>(&[]).unwrap();
    let batches: Vec<RecordBatch> = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    concat_batches(&batches[0].schema(), &batches).unwrap()
}

/// An index search on a cold cache reserves what its index loads read and
/// releases it once loaded, so it peaks above the same search on a warm cache.
#[rstest]
#[case::vector(IndexSearch::Vector)]
#[case::btree(IndexSearch::BTree)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_index_loads_reserve_the_budget(#[case] search: IndexSearch) {
    let dir = TempStrDir::default();
    write_indexed(&dir).await;
    let expected = index_search(&open(&dir, None).await, search).await;
    assert!(expected.num_rows() > 0);
    let dataset = open(&dir, Some(64 * 1024 * 1024)).await;
    let budget = dataset.session.read_budget().unwrap();
    let mut peaks = Vec::new();
    for _ in 0..2 {
        budget.reset_peak();
        assert_eq!(index_search(&dataset, search).await, expected);
        assert_released(&dataset);
        peaks.push(budget.stats().peak_reserved_bytes);
    }
    assert!(peaks[0] > peaks[1], "cold peak {peaks:?} not above warm");
}

/// A cold index load waits for room: while another query holds the
/// overdraft and the budget is full, a vector search cannot load its
/// partitions; once that query releases, it finishes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_index_load_waits_for_the_budget() {
    let dir = TempStrDir::default();
    write_indexed(&dir).await;
    let expected = index_search(&open(&dir, None).await, IndexSearch::Vector).await;
    let limit = 64 * 1024;
    let dataset = open(&dir, Some(limit)).await;
    let budget = dataset.session.read_budget().unwrap();
    let held = budget
        .reader(QueryId::unique())
        .admission()
        .reserve(limit + 1)
        .await;
    held.resize(limit);
    let search = tokio::spawn({
        let dataset = dataset.clone();
        async move { index_search(&dataset, IndexSearch::Vector).await }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(!search.is_finished(), "an index load ignored the budget");
    drop(held);
    let actual = tokio::time::timeout(Duration::from_secs(60), search)
        .await
        .expect("the index load stayed blocked after the budget freed")
        .unwrap();
    assert_eq!(actual, expected);
    assert_released(&dataset);
}

/// A read dropped after its first batch releases everything it reserved,
/// including what its spawned chunk reads held, even while its plan is kept.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_cancelled_budgeted_reads_release(
    #[values(Shape::FixedBinary, Shape::LargeBinary)] shape: Shape,
) {
    let dir = TempStrDir::default();
    write(&dir, LanceFileVersion::V2_2, shape, 4, None).await;
    let dataset = open(&dir, Some(4 * 1024 * 1024)).await;
    let mut scanner = dataset.scan();
    scanner.batch_size(16).project(&["id", "value"]).unwrap();
    let mut stream = scanner.try_into_stream().await.unwrap();
    stream.try_next().await.unwrap().unwrap();
    drop(stream);
    let config = LanceScanConfig {
        batch_size: 16,
        ..Default::default()
    };
    let mut stream = fragment_scan(&dataset, None, config)
        .execute(0, context())
        .unwrap();
    stream.next().await.unwrap().unwrap();
    drop(stream);
    let mut options = FilteredReadOptions::basic_full_read(&dataset).with_batch_size(16);
    options.threading_mode = FilteredReadThreadingMode::OnePartitionMultipleThreads(4);
    let kept = FilteredReadExec::try_new(dataset.clone(), options, None).unwrap();
    let mut stream = kept.execute(0, context()).unwrap();
    stream.try_next().await.unwrap().unwrap();
    drop(stream);
    assert_released_eventually(&dataset).await;
    drop(kept);
}

/// A read whose data reads fail releases everything it reserved.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_failed_budgeted_reads_release() {
    let dir = TempStrDir::default();
    let rows = write(&dir, LanceFileVersion::V2_2, Shape::FixedBinary, 4, None).await;
    // Opens succeed; data reads fail once the read is under way.
    let store = Arc::new(FailingProxyStore::new());
    for method in ["get_opts", "get_ranges"] {
        store.fail_after_n(method, "/data/", 8, "injected read failure");
    }
    let dataset = DatasetBuilder::from_uri(format!("file-object-store://{dir}"))
        .with_store_params(ObjectStoreParams {
            object_store_wrapper: Some(store),
            ..Default::default()
        })
        .with_session(Arc::new(
            Session::default().with_read_memory_limit(4 * 1024 * 1024),
        ))
        .load()
        .await
        .unwrap();
    let dataset = Arc::new(dataset);
    let mut scanner = dataset.scan();
    scanner.batch_size(16).project(&["id", "value"]).unwrap();
    assert!(
        scanner.try_into_batch().await.is_err(),
        "the injected failure must reach the reader"
    );
    let take = dataset
        .take(
            &(0..rows).step_by(7).collect::<Vec<_>>(),
            ProjectionRequest::from_columns(["value"], dataset.schema()),
        )
        .await;
    assert!(take.is_err());
    assert_released_eventually(&dataset).await;
}

/// A scan its consumer has paused does not stop another query whose reads
/// fit, nor a `Dataset::take` while the paused scan holds the overdraft, since
/// a take never waits.
#[rstest]
#[case::reads_that_fit(64 * 1024 * 1024, &READS)]
#[case::take_beside_the_overdraft(1, &[Read::Take])]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_paused_scan_does_not_block_other_reads(#[case] limit: u64, #[case] reads: &[Read]) {
    let dir = TempStrDir::default();
    let rows = write(&dir, LanceFileVersion::V2_2, Shape::FixedBinary, 4, None).await;
    let unbudgeted = open(&dir, None).await;
    let dataset = open(&dir, Some(limit)).await;
    let mut scanner = dataset.scan();
    scanner.batch_size(16).project(&["id", "value"]).unwrap();
    let mut paused = scanner.try_into_stream().await.unwrap();
    let mut scanned = paused.try_next().await.unwrap().unwrap().num_rows();
    for &kind in reads {
        let actual = within(120, read(&dataset, kind, 1024, rows)).await;
        assert_eq!(
            actual,
            read(&unbudgeted, kind, 1024, rows).await,
            "{kind:?}"
        );
    }
    while let Some(batch) = paused.try_next().await.unwrap() {
        scanned += batch.num_rows();
    }
    assert_eq!(scanned as u64, rows);
    assert_released(&dataset);
}

/// Partitions admit their chunks independently: a partition holding a chunk
/// it has not emitted never blocks another partition, even while another
/// query fills the budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_budgeted_partitions_never_wait_for_each_other() {
    let dir = TempStrDir::default();
    let rows = write(&dir, LanceFileVersion::V2_2, Shape::Int, 8, None).await;
    let expected = read(&open(&dir, None).await, Read::Scan, 1024, rows).await;
    let limit = 16 * 1024;
    let dataset = open(&dir, Some(limit)).await;
    let other_query = dataset
        .session
        .read_budget()
        .unwrap()
        .reader(QueryId::unique())
        .charge(limit);
    // Chunks of many batches each.
    let mut options = FilteredReadOptions::basic_full_read(&dataset).with_batch_size(256);
    options.threading_mode = FilteredReadThreadingMode::MultiplePartitions(2);
    let plan = FilteredReadExec::try_new(dataset.clone(), options, None).unwrap();
    let context = context();
    let mut stalled = plan.execute(0, context.clone()).unwrap();
    let mut batches = vec![stalled.try_next().await.unwrap().unwrap()];
    let drained = within(
        30,
        plan.execute(1, context).unwrap().try_collect::<Vec<_>>(),
    )
    .await
    .unwrap();
    batches.extend(drained);
    batches.extend(stalled.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(by_id(batches).unwrap(), expected);
    drop(other_query);
    assert_released(&dataset);
}

/// Deleted rows, a column added in a second data file per fragment, stable row
/// ids and system columns, with one fragment read ahead at a time.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_budgeted_reads_of_evolved_datasets() {
    let dir = TempStrDir::default();
    let (generator, rows) = Shape::FixedBinary.generator();
    let mut dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", generator)
        .into_dataset_with_params(
            &dir,
            FragmentCount::from(4),
            FragmentRowCount::from((rows / 4) as u32),
            Some(WriteParams {
                max_rows_per_file: (rows / 4) as usize,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    dataset.delete("id % 5 = 0").await.unwrap();
    dataset
        .add_columns(
            crate::dataset::NewColumnTransform::SqlExpressions(vec![(
                "extra".to_string(),
                "id * 2".to_string(),
            )]),
            None,
            None,
        )
        .await
        .unwrap();
    let scan = |dataset: Arc<Dataset>| async move {
        let mut scanner = dataset.scan();
        scanner
            .project(&["id", "value", "extra"])
            .unwrap()
            .with_row_id()
            .with_row_address()
            .fragment_readahead(1)
            .filter("id % 3 = 0")
            .unwrap();
        by_id(vec![scanner.try_into_batch().await.unwrap()])
    };
    // A whole-fragment scan that keeps deleted rows, with a null row id.
    let deletions_null = |dataset: Arc<Dataset>| async move {
        let config = LanceScanConfig {
            with_row_id: true,
            with_make_deletions_null: true,
            ..Default::default()
        };
        by_id(collect(&fragment_scan(&dataset, None, config)).await)
    };
    let unbudgeted = open(&dir, None).await;
    let expected = (
        scan(unbudgeted.clone()).await,
        deletions_null(unbudgeted.clone()).await,
        read_all(&unbudgeted, 1024, rows).await,
    );
    for limit in [1, 4 * 1024 * 1024] {
        let dataset = open(&dir, Some(limit)).await;
        let actual = within(300, async {
            (
                scan(dataset.clone()).await,
                deletions_null(dataset.clone()).await,
                read_all(&dataset, 1024, rows).await,
            )
        })
        .await;
        assert_eq!(actual, expected, "limit {limit}");
        assert_released(&dataset);
    }
}

/// The estimate a budgeted read reserves covers what the read loads, for every
/// file version, column shape and page size, whole fragments and parts of them.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_estimate_covers_bytes_read(
    #[values(
        LanceFileVersion::V2_0,
        LanceFileVersion::V2_1,
        LanceFileVersion::V2_2,
        LanceFileVersion::V2_3
    )]
    version: LanceFileVersion,
    #[values(
        Shape::Int,
        Shape::Utf8,
        Shape::VarBinary,
        Shape::FixedBinary,
        Shape::ListOfBinary,
        Shape::Vector,
        Shape::Struct,
        Shape::LargeBinary,
        Shape::LowCardinality,
        Shape::AllNull
    )]
    shape: Shape,
    #[values(None, Some(64 * 1024), Some(1 << 30))] max_page_bytes: Option<u64>,
) {
    let dir = TempStrDir::default();
    let rows = write(&dir, version, shape, 1, max_page_bytes).await;
    let schema = open(&dir, None).await.schema().project(&["value"]).unwrap();
    let chunk = rows / 16;
    let cases: Vec<(&str, Vec<Range<u64>>)> = vec![
        ("whole", vec![0..rows]),
        ("first chunk", vec![0..chunk]),
        ("middle chunk", vec![rows / 2..rows / 2 + chunk]),
        (
            "scattered",
            (0..rows - 2)
                .step_by((rows / 40).max(3) as usize)
                .map(|row| row..row + 2)
                .collect(),
        ),
    ];
    for (name, ranges) in cases {
        // A fresh session and reader per case, so no case finds metadata or
        // pages another loaded already cached.
        let dataset = open(&dir, None).await;
        let fragment = dataset.get_fragments().remove(0);
        let reader = fragment
            .open(&schema, FragReadConfig::default())
            .await
            .unwrap();
        dataset.object_store.io_stats_incremental();
        let estimate = reader
            .estimate_read(ranges.clone().into())
            .await
            .unwrap()
            .bytes as f64;
        // Estimates come from metadata the reader already holds.
        assert_eq!(
            dataset.object_store.io_stats_incremental().read_iops,
            0,
            "{name}: estimating issued reads"
        );
        let _: Vec<RecordBatch> = reader
            .read_ranges(ranges.into(), 1024)
            .await
            .unwrap()
            .buffered(4)
            .try_collect()
            .await
            .unwrap();
        let read = dataset.object_store.io_stats_incremental().read_bytes as f64;
        assert!(
            read <= 1.5 * estimate,
            "{name}: reads {read} bytes but estimates {estimate}"
        );
        // Stable formats are estimated closely enough not to hold back reads.
        if matches!(version, LanceFileVersion::V2_1 | LanceFileVersion::V2_2) {
            assert!(
                estimate <= 1.5 * read + 4096.0,
                "{name}: estimates {estimate} bytes but reads {read}"
            );
        }
    }
}
