// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    RecordBatchReader,
    types::{UInt32Type, UInt64Type},
};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::{physical_plan::SendableRecordBatchStream, scalar::ScalarValue};
use futures::{FutureExt, TryStreamExt};
use lance::dataset::WriteParams;
use lance::dataset::scanner::{ColumnOrdering, OrderedIndexMode, Scanner};
use lance::index::DatasetIndexExt;
use lance::{Dataset, io::ObjectStore};
use lance_core::ROW_ADDR;
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempStrDir;
use lance_datafusion::utils::reader_to_stream;
use lance_datagen::{BatchCount, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::scalar::{
    BuiltinIndexType, IndexStore, SargableQuery, ScalarIndex, ScalarIndexParams, SearchResult,
    btree::{DEFAULT_BTREE_BATCH_SIZE, train_btree_index},
    lance_format::LanceIndexStore,
    registry::{ScalarIndexPlugin, VALUE_COLUMN_NAME},
};
use lance_index::{metrics::NoOpMetricsCollector, scalar::btree::BTreeIndexPlugin};
use lance_select::RowSetOps;
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

struct BenchmarkFixture {
    _datadir: TempStrDir,
    index_store: Arc<dyn IndexStore>,
    baseline_dataset: Arc<Dataset>,
}

fn test_data(value_column: &str, row_column: &str) -> impl RecordBatchReader {
    gen_batch()
        .col(value_column, array::step::<UInt32Type>())
        .col(row_column, array::step::<UInt64Type>())
        .into_reader_rows(RowCount::from(1024), BatchCount::from(100 * 1024))
}

fn test_data_stream() -> SendableRecordBatchStream {
    reader_to_stream(Box::new(test_data(VALUE_COLUMN_NAME, ROW_ADDR)))
}

impl BenchmarkFixture {
    fn test_store(tempdir: &TempStrDir) -> Arc<dyn IndexStore> {
        let test_path = tempdir.as_str();
        let (object_store, test_path) = ObjectStore::from_uri(test_path)
            .now_or_never()
            .unwrap()
            .unwrap();
        Arc::new(LanceIndexStore::new(
            object_store,
            test_path,
            Arc::new(LanceCache::no_cache()),
        ))
    }

    async fn write_baseline_data(tempdir: &TempStrDir) -> Arc<Dataset> {
        let test_path = tempdir.as_str();
        Arc::new(
            Dataset::write(test_data("values", "row_ids"), test_path, None)
                .await
                .unwrap(),
        )
    }

    async fn train_scalar_index(index_store: &Arc<dyn IndexStore>) {
        train_btree_index(
            test_data_stream(),
            index_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            None,
            None,
        )
        .await
        .unwrap();
    }

    async fn open() -> Self {
        let tempdir = TempStrDir::default();
        let index_store = Self::test_store(&tempdir);
        let baseline_dataset = Self::write_baseline_data(&tempdir).await;
        Self::train_scalar_index(&index_store).await;

        Self {
            _datadir: tempdir,
            index_store,
            baseline_dataset,
        }
    }
}

async fn baseline_equality_search(fixture: &BenchmarkFixture) {
    let mut stream = fixture
        .baseline_dataset
        .scan()
        .filter("values == 10000")
        .unwrap()
        .with_row_id()
        .try_into_stream()
        .await
        .unwrap();
    let mut num_rows = 0;
    while let Some(batch) = stream.try_next().await.unwrap() {
        num_rows += batch.num_rows();
    }
    assert_eq!(num_rows, 1);
}

async fn warm_indexed_equality_search(index: &dyn ScalarIndex) {
    let result = index
        .search(
            &SargableQuery::Equals(ScalarValue::UInt32(Some(10000))),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
    let SearchResult::Exact(row_ids) = result else {
        panic!("Expected exact results")
    };
    assert_eq!(row_ids.true_rows().len(), Some(1));
}

async fn baseline_inequality_search(fixture: &BenchmarkFixture) {
    let mut stream = fixture
        .baseline_dataset
        .scan()
        .filter("values >= 50000000")
        .unwrap()
        .with_row_id()
        .try_into_stream()
        .await
        .unwrap();
    let mut num_rows = 0;
    while let Some(batch) = stream.try_next().await.unwrap() {
        num_rows += batch.num_rows();
    }
    // 100Mi - 50M = 54,857,600
    assert_eq!(num_rows, 54857600);
}

async fn warm_indexed_inequality_search(index: &dyn ScalarIndex) {
    let result = index
        .search(
            &SargableQuery::Range(
                std::ops::Bound::Included(ScalarValue::UInt32(Some(50_000_000))),
                std::ops::Bound::Unbounded,
            ),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
    let SearchResult::Exact(row_ids) = result else {
        panic!("Expected exact results")
    };

    // 100Mi - 50M = 54,857,600
    assert_eq!(row_ids.true_rows().len(), Some(54857600));
}

async fn warm_indexed_isin_search(index: &dyn ScalarIndex) {
    let result = index
        .search(
            &SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(10000)),
                ScalarValue::UInt32(Some(50000000)),
                ScalarValue::UInt32(Some(150000000)), // Not found
                ScalarValue::UInt32(Some(287123)),
            ]),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
    let SearchResult::Exact(row_ids) = result else {
        panic!("Expected exact results")
    };

    // Only 3 because 150M is not in dataset
    assert_eq!(row_ids.true_rows().len(), Some(3));
}

async fn ordered_limit_dataset(tempdir: &TempStrDir) -> Dataset {
    let data = gen_batch()
        .col("ts", array::rand::<UInt64Type>())
        .col("id", array::step::<UInt64Type>())
        .into_reader_rows(RowCount::from(100_000), BatchCount::from(100));
    let params = WriteParams {
        max_rows_per_file: 1_000_000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(data, tempdir.as_str(), Some(params))
        .await
        .unwrap();
    dataset
        .create_index(
            &["ts"],
            IndexType::BTree,
            Some("ts_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();
    dataset
}

async fn ordered_limit(dataset: &Dataset, configure: impl Fn(&mut Scanner)) {
    let mut scanner = dataset.scan();
    scanner
        .project(&["id"])
        .unwrap()
        .order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".into())]))
        .unwrap()
        .limit(Some(100), None)
        .unwrap();
    configure(&mut scanner);
    assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 100);
}

fn bench_baseline(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let fixture = rt.block_on(BenchmarkFixture::open());

    c.bench_function("baseline_equality", |b| {
        b.iter(|| rt.block_on(baseline_equality_search(&fixture)));
    });

    c.bench_function("baseline_inequality", |b| {
        b.iter(|| rt.block_on(baseline_inequality_search(&fixture)));
    });
}

fn bench_warm_indexed(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(BenchmarkFixture::open());
    let details =
        prost_types::Any::from_msg(&lance_index::pbold::BTreeIndexDetails::default()).unwrap();

    let index = rt
        .block_on(BTreeIndexPlugin.load_index(
            fixture.index_store.clone(),
            &details,
            lance_index::scalar::btree::BTREE_INDEX_VERSION,
            None,
            &LanceCache::no_cache(),
        ))
        .unwrap();

    c.bench_function("windexed_equality", |b| {
        b.iter(|| rt.block_on(warm_indexed_equality_search(index.as_ref())))
    });

    c.bench_function("windexed_inequality", |b| {
        b.iter(|| rt.block_on(warm_indexed_inequality_search(index.as_ref())))
    });

    c.bench_function("windexed_is_in", |b| {
        b.iter(|| rt.block_on(warm_indexed_isin_search(index.as_ref())))
    });
}

fn bench_ordered_limit(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let tempdir = TempStrDir::default();
    let dataset = rt.block_on(ordered_limit_dataset(&tempdir));
    let mut group = c.benchmark_group("ordered_limit");
    group.bench_function("ordered_index_auto", |b| {
        b.iter(|| rt.block_on(ordered_limit(&dataset, |_| {})));
    });
    group.bench_function("ordered_index_off", |b| {
        b.iter(|| {
            rt.block_on(ordered_limit(&dataset, |scanner| {
                scanner.ordered_index(OrderedIndexMode::Off);
            }))
        });
    });
    group.bench_function("declined_by_filter", |b| {
        b.iter(|| {
            rt.block_on(ordered_limit(&dataset, |scanner| {
                scanner.filter("id >= 0").unwrap();
            }))
        });
    });
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_baseline, bench_warm_indexed, bench_ordered_limit);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_baseline, bench_warm_indexed, bench_ordered_limit);

criterion_main!(benches);
