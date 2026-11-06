// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reads under a session read memory limit return what unlimited reads return,
//! for every file version, column shape, page size, limit and batch size, alone
//! and concurrently, and release everything they reserve, including when they
//! are cancelled or fail.

use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
use arrow_array::RecordBatch;
use arrow_array::types::{Float32Type, Int32Type};
use arrow_schema::{DataType, Field, Fields};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::tempfile::TempStrDir;
use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
use lance_datagen::{ArrayGenerator, ByteCount, array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_file::writer::FileWriterOptions;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
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
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

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

async fn open(dir: &str, limit: Option<u64>) -> Arc<Dataset> {
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
            scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap()
        }
        Read::Take => {
            let mut rows: Vec<u64> = (0..rows).step_by(13).collect();
            rows.reverse();
            rows.extend_from_slice(&[0, 0, rows[0]]);
            let projection = ProjectionRequest::from_columns(["id", "value"], dataset.schema());
            // In request order: compared as is, not sorted.
            return dataset.take(&rows, projection).await.unwrap();
        }
        Read::FragmentScan => {
            let scan = LanceScanExec::new(
                dataset.clone(),
                dataset.fragments().clone(),
                None,
                Arc::new(dataset.schema().project(&["id", "value"]).unwrap()),
                LanceScanConfig {
                    batch_size,
                    ..Default::default()
                },
            );
            scan.execute(0, context())
                .unwrap()
                .try_collect()
                .await
                .unwrap()
        }
        Read::TakeExec => {
            let scan = LanceScanExec::new(
                dataset.clone(),
                dataset.fragments().clone(),
                None,
                Arc::new(dataset.schema().project(&["id"]).unwrap()),
                LanceScanConfig {
                    batch_size,
                    with_row_address: true,
                    ..Default::default()
                },
            );
            let value = dataset
                .empty_projection()
                .union_column("value", lance_core::datatypes::OnMissing::Error)
                .unwrap();
            let take = TakeExec::try_new(dataset.clone(), Arc::new(scan), value)
                .unwrap()
                .unwrap();
            let batches: Vec<RecordBatch> = take
                .execute(0, context())
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            batches
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

fn assert_released(dataset: &Dataset) {
    let budget = dataset.session.read_budget().unwrap();
    assert_eq!(budget.reserved_bytes(), 0, "a read kept a reservation");
}

/// Waits for spawned reads of dropped streams to wind down and release.
async fn assert_released_eventually(dataset: &Dataset) {
    let budget = dataset.session.read_budget().unwrap();
    for _ in 0..1000 {
        if budget.reserved_bytes() == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{} bytes stayed reserved", budget.reserved_bytes());
}

/// Every read path returns exactly what it returns without a limit, under a
/// limit every unit overdraws (1 byte), one that its read-ahead waits on
/// (4 MiB) and one that rarely binds, for any batch size, alone and with all
/// paths at once.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_budgeted_reads_match_unbudgeted(
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
        Shape::LargeBinary
    )]
    shape: Shape,
) {
    let dir = TempStrDir::default();
    let rows = write(&dir, version, shape, 4, None).await;
    let expected = read_all(&open(&dir, None).await, 1024, rows).await;
    let check = async {
        for limit in [1, 4 * 1024 * 1024, 64 * 1024 * 1024] {
            let dataset = open(&dir, Some(limit)).await;
            for batch_size in [3, 1024, 65536] {
                assert_eq!(
                    read_all(&dataset, batch_size, rows).await,
                    expected,
                    "limit {limit}, batch size {batch_size}"
                );
                assert_released(&dataset);
            }
            let concurrent = futures::future::join_all(READS.iter().map(|kind| {
                let dataset = dataset.clone();
                async move { read(&dataset, *kind, 1024, rows).await }
            }))
            .await;
            assert_eq!(concurrent, expected, "limit {limit}, concurrent");
            assert_released(&dataset);
        }
    };
    tokio::time::timeout(Duration::from_secs(600), check)
        .await
        .expect("a budgeted read hung");
}

/// Many queries at once, each a different read path, under a limit far below
/// what they want together.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_many_concurrent_budgeted_reads(
    #[values(Shape::FixedBinary, Shape::LargeBinary, Shape::Struct)] shape: Shape,
) {
    let dir = TempStrDir::default();
    let rows = write(&dir, LanceFileVersion::V2_2, shape, 8, None).await;
    let expected = read_all(&open(&dir, None).await, 1024, rows).await;
    let dataset = open(&dir, Some(4 * 1024 * 1024)).await;
    let reads = (0..64).map(|query| {
        let (dataset, kind) = (dataset.clone(), READS[query % READS.len()]);
        async move { (query % READS.len(), read(&dataset, kind, 1024, rows).await) }
    });
    let results = tokio::time::timeout(Duration::from_secs(600), futures::future::join_all(reads))
        .await
        .expect("concurrent budgeted reads hung");
    for (kind, result) in results {
        assert_eq!(result, expected[kind], "{:?}", READS[kind]);
    }
    assert_released(&dataset);
}

/// Tiny pages and pages larger than the limit, and thousands of tiny fragments.
#[rstest]
#[case::tiny_pages(Shape::Utf8, 4, Some(4096))]
#[case::huge_pages(Shape::FixedBinary, 4, Some(1 << 30))]
#[case::many_fragments(Shape::FixedBinary, 2000, None)]
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
    let actual = tokio::time::timeout(Duration::from_secs(600), read_all(&dataset, 1024, rows))
        .await
        .expect("a budgeted read hung");
    assert_eq!(actual, expected);
    assert_released(&dataset);
}

/// Index-driven reads: a scalar-index filter loads scattered rows, and a
/// vector search refines its candidates with a take.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_budgeted_index_reads() {
    let dir = TempStrDir::default();
    let rows = write(&dir, LanceFileVersion::V2_2, Shape::Vector, 4, None).await;
    let mut dataset = (*open(&dir, None).await).clone();
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
    let query = arrow_array::Float32Array::from(vec![0.5; 128]);
    let search = |dataset: Arc<Dataset>| {
        let query = query.clone();
        async move {
            let mut filtered = dataset.scan();
            filtered
                .filter("id < 50 OR id = 4000 OR id > 8100")
                .unwrap()
                .project(&["id", "value"])
                .unwrap();
            let filtered = by_id(
                filtered
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap(),
            );
            let mut knn = dataset.scan();
            knn.nearest("value", &query, 10)
                .unwrap()
                .refine(4)
                .project(&["id"])
                .unwrap();
            let knn = by_id(
                knn.try_into_stream()
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap(),
            );
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
        let actual = tokio::time::timeout(Duration::from_secs(300), search(dataset.clone()))
            .await
            .expect("a budgeted index read hung");
        assert_eq!(actual, expected, "limit {limit}");
        assert_released(&dataset);
    }
}

/// A read dropped after its first batch releases everything it reserved,
/// including what its spawned chunk reads held.
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
    let scan = LanceScanExec::new(
        dataset.clone(),
        dataset.fragments().clone(),
        None,
        Arc::new(dataset.schema().clone()),
        LanceScanConfig {
            batch_size: 16,
            ..Default::default()
        },
    );
    let mut stream = scan.execute(0, context()).unwrap();
    stream.next().await.unwrap().unwrap();
    drop(stream);
    assert_released_eventually(&dataset).await;
}

/// A read whose data reads fail releases everything it reserved.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_failed_budgeted_reads_release() {
    let dir = TempStrDir::default();
    let rows = write(&dir, LanceFileVersion::V2_2, Shape::FixedBinary, 4, None).await;
    let mut dataset = (*open(&dir, Some(4 * 1024 * 1024)).await).clone();
    // Opens succeed; data reads fail once the read is under way.
    let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut policy = ProxyObjectStorePolicy::new();
    policy.set_before_policy(
        "fail_data_reads",
        Arc::new(move |method, path| {
            let is_data = path.as_ref().contains("/data/") && method.starts_with("get");
            if is_data && reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed) > 8 {
                return Err(object_store::Error::Generic {
                    store: "test",
                    source: "injected read failure".into(),
                }
                .into());
            }
            Ok(())
        }),
    );
    let mut store = dataset.object_store.as_ref().clone();
    store.inner = Arc::new(ProxyObjectStore::new(
        store.inner.clone(),
        Arc::new(Mutex::new(policy)),
    ));
    dataset.object_store = Arc::new(store);
    let dataset = Arc::new(dataset);
    let mut scanner = dataset.scan();
    scanner.batch_size(16).project(&["id", "value"]).unwrap();
    let result: lance_core::Result<Vec<RecordBatch>> =
        async { scanner.try_into_stream().await?.try_collect().await }.await;
    assert!(
        result.is_err(),
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
        Shape::LargeBinary
    )]
    shape: Shape,
    #[values(None, Some(64 * 1024), Some(1 << 30))] max_page_bytes: Option<u64>,
) {
    let dir = TempStrDir::default();
    let rows = write(&dir, version, shape, 1, max_page_bytes).await;
    let dataset = open(&dir, None).await;
    let fragment = dataset.get_fragments().remove(0);
    let schema = dataset.schema().project(&["value"]).unwrap();
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
        // A fresh reader per case, so no case reads pages another loaded.
        let reader = fragment
            .open(&schema, FragReadConfig::default())
            .await
            .unwrap();
        let estimate = reader
            .estimate_read_bytes(ranges.clone().into())
            .await
            .unwrap() as f64;
        dataset.object_store.io_stats_incremental();
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
