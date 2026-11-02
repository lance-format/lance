// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::IVFIndex;
use super::native_resident_tests::open_with_session;
use super::native_storage_tests::*;
use super::*;
use crate::index::vector::ivf::v2::rabit_scan::RabitCacheLayout;
use crate::session::Session;
use async_trait::async_trait;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::{FutureExt, StreamExt};
use lance_core::Error;
use lance_core::cache::QuickCacheBackend;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::storage::{StagePruneCounts, select_full_survivors};
use lance_index::vector::storage::{
    ExIndex, GatherPlan, GatheredRabitEx, IndexFileKey, OriginLatencyClass, PlaneSource,
};
use lance_index::vector::{PartitionSearchControl, VECTOR_RESULT_SCHEMA};
use lance_table::transaction::{Operation, RewriteGroup, Transaction};

fn lazy_index(index: &Arc<dyn VectorIndex>) -> &IVFIndex<FlatIndex, RabitQuantizer> {
    index
        .as_any()
        .downcast_ref::<IVFIndex<FlatIndex, RabitQuantizer>>()
        .unwrap()
}

#[derive(Default)]
struct RowBudget {
    limit: usize,
    rows: AtomicUsize,
    batches: AtomicUsize,
    stopped: AtomicBool,
}

impl PartitionSearchControl for RowBudget {
    fn should_stop(&self) -> bool {
        self.stopped.load(Ordering::Relaxed) || self.rows.load(Ordering::Relaxed) >= self.limit
    }

    fn record_batch(&self, batch: &RecordBatch) {
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(batch.num_rows(), Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
struct PartitionScanFactory {
    index: Arc<dyn VectorIndex>,
    calls: Arc<AtomicUsize>,
    produced: Arc<AtomicUsize>,
}

impl super::super::rabit_scan::RabitScanFactory for PartitionScanFactory {
    fn create(
        &self,
        _: &IndexFileKey,
        _: OriginLatencyClass,
    ) -> Result<Arc<dyn super::super::rabit_scan::RabitScan>> {
        Ok(Arc::new(self.clone()))
    }
}

#[async_trait]
impl super::super::rabit_scan::RabitScan for PartitionScanFactory {
    async fn search(
        &self,
        _: Arc<super::super::rabit_scan::RabitScanContext>,
    ) -> Result<Option<RecordBatch>> {
        Err(Error::internal(
            "controlled probes called the global-heap strategy",
        ))
    }

    fn search_partition_stream(
        &self,
        context: Arc<super::super::rabit_scan::RabitScanContext>,
    ) -> Result<Option<SendableRecordBatchStream>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let produced = self.produced.clone();
        let index = self.index.clone();
        let stream = futures::stream::iter(context.probes.clone()).then(move |rank| {
            let context = context.clone();
            let produced = produced.clone();
            let index = index.clone();
            async move {
                let mut query = context.query.clone();
                query.dist_q_c = context.centroid_distances.value(rank);
                let batch = index
                    .search_in_partition(
                        context.partitions.value(rank) as usize,
                        &query,
                        context.pre_filter.clone(),
                        context.metrics.as_ref(),
                    )
                    .await?;
                produced.fetch_add(1, Ordering::Relaxed);
                Ok(batch)
            }
        });
        Ok(Some(Box::pin(RecordBatchStreamAdapter::new(
            VECTOR_RESULT_SCHEMA.clone(),
            stream,
        ))))
    }
}

#[rstest]
#[case::all_probes(0, usize::MAX, false)]
#[case::late_probes(1, 1, false)]
#[case::already_stopped(1, 0, false)]
#[case::compacted(1, 1, true)]
#[tokio::test]
async fn controlled_native_strategy_preserves_partition_batches(
    #[case] start: usize,
    #[case] limit: usize,
    #[case] compact: bool,
    #[values(false, true)] empty_filter: bool,
) {
    let dir = TempStrDir::default();
    let (mut dataset, batch) = native_fixture(dir.as_str()).await;
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    if compact {
        dataset.delete("i % 7 = 0").await.unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    }
    let mut query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 4, 4);
    query.refine_factor = Some(3);
    let ordinary = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let strategy = Arc::new(PartitionScanFactory {
        index: ordinary.clone(),
        calls: Arc::default(),
        produced: Arc::default(),
    });
    let configured = open_with_session(
        dir.as_str(),
        Session::default().with_rabit_scan_factory(strategy.clone()),
    )
    .await;
    let selected = configured
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let old_factory = Arc::new(RecordingScanFactory {
        opens: Arc::default(),
        searches: Arc::default(),
        behavior: TestScanBehavior::SearchError,
        cache_layout: RabitCacheLayout::Partition,
    });
    let fallback = open_with_session(
        dir.as_str(),
        Session::default().with_rabit_scan_factory(old_factory.clone()),
    )
    .await;
    let fallback_index = fallback
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let filter: Arc<dyn PreFilter> = if empty_filter {
        Arc::new(DatasetPreFilter::new_with_filter_future(
            Arc::new(dataset.clone()),
            &dataset.load_indices().await.unwrap(),
            Some(async { Ok(Arc::new(RowAddrMask::AllowList(RowAddrTreeMap::default()))) }.boxed()),
        ))
    } else {
        Arc::new(NoFilter)
    };
    let mut outputs = Vec::new();
    for index in [ordinary, selected, fallback_index] {
        let budget = Arc::new(RowBudget {
            limit,
            ..Default::default()
        });
        let (partitions, distances) = index.find_partitions(&query).unwrap();
        let batches = index
            .search_partitions(
                query.clone(),
                Arc::new(partitions),
                Arc::new(distances),
                start,
                4,
                filter.clone(),
                Some(budget.clone()),
                Arc::new(NoOpMetricsCollector),
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count = if limit == 0 {
            0
        } else if empty_filter || limit == usize::MAX {
            4 - start
        } else {
            1
        };
        assert_eq!(batches.len(), count);
        assert_eq!(budget.batches.load(Ordering::Relaxed), count);
        assert_eq!(
            budget.rows.load(Ordering::Relaxed),
            batches.iter().map(RecordBatch::num_rows).sum::<usize>()
        );
        outputs.push(batches.iter().map(result_bits).collect::<Vec<_>>());
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[0], outputs[2]);
    assert_eq!(old_factory.searches.load(Ordering::Relaxed), 0);
    assert_eq!(
        strategy.calls.load(Ordering::Relaxed),
        usize::from(limit != 0)
    );
    assert_eq!(strategy.produced.load(Ordering::Relaxed), outputs[1].len());
    if start == 0 && !empty_filter {
        let truth =
            super::ground_truth(&dataset, "vector", query.key.as_ref(), 4, DistanceType::L2).await;
        assert!(
            outputs[1]
                .iter()
                .flat_map(|(ids, _)| ids)
                .filter(|id| truth.contains(id))
                .count()
                >= 2
        );
    }
}

struct StreamDrop(Arc<AtomicBool>);

impl Drop for StreamDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy)]
enum PartitionStreamCase {
    EmptyBatch,
    Short,
    Extra,
    OverBudget,
    BatchError,
    BatchSchema,
    StreamSchema,
    StopOnPoll,
    StopWithError,
}

#[rstest]
#[case::empty_batch(PartitionStreamCase::EmptyBatch, 2, 1, None)]
#[case::short(
    PartitionStreamCase::Short,
    1,
    1,
    Some("ended after 1 batches; expected 2")
)]
#[case::extra(PartitionStreamCase::Extra, 2, 2, Some("more than 2 batches"))]
#[case::over_budget(
    PartitionStreamCase::OverBudget,
    0,
    0,
    Some("exceeding candidate limit 1")
)]
#[case::batch_error(
    PartitionStreamCase::BatchError,
    1,
    1,
    Some("injected partition error")
)]
#[case::batch_schema(PartitionStreamCase::BatchSchema, 0, 0, Some("unexpected schema"))]
#[case::stream_schema(
    PartitionStreamCase::StreamSchema,
    0,
    0,
    Some("partition stream schema")
)]
#[case::stop_during_poll(PartitionStreamCase::StopOnPoll, 0, 0, None)]
#[case::error_during_stop(
    PartitionStreamCase::StopWithError,
    0,
    0,
    Some("injected partition error")
)]
#[tokio::test]
async fn controlled_partition_stream_enforces_contract(
    #[case] scenario: PartitionStreamCase,
    #[case] expected_batches: usize,
    #[case] expected_rows: usize,
    #[case] expected_error: Option<&str>,
) {
    let two = RecordBatch::try_new(
        VECTOR_RESULT_SCHEMA.clone(),
        vec![
            Arc::new(Float32Array::from(vec![0.0, 1.0])),
            Arc::new(UInt64Array::from(vec![1, 2])),
        ],
    )
    .unwrap();
    let one = two.slice(0, 1);
    let failure = || {
        Err(DataFusionError::from(Error::internal(
            "injected partition error",
        )))
    };
    let items = match scenario {
        PartitionStreamCase::EmptyBatch => vec![Ok(one.slice(0, 0)), Ok(one)],
        PartitionStreamCase::Short => vec![Ok(one)],
        PartitionStreamCase::Extra => vec![Ok(one.clone()), Ok(one.clone()), Ok(one)],
        PartitionStreamCase::OverBudget => vec![Ok(two)],
        PartitionStreamCase::BatchError => vec![Ok(one.clone()), failure(), Ok(one)],
        PartitionStreamCase::BatchSchema => {
            vec![Ok(RecordBatch::new_empty(Arc::new(Schema::empty())))]
        }
        PartitionStreamCase::StopWithError => vec![failure()],
        PartitionStreamCase::StreamSchema | PartitionStreamCase::StopOnPoll => vec![Ok(one)],
    };
    let control = Arc::new(RowBudget {
        limit: usize::MAX,
        ..Default::default()
    });
    let producer_control = control.clone();
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = StreamDrop(dropped.clone());
    let source = futures::stream::iter(items).map(move |item| {
        let _keep_alive = &guard;
        if matches!(
            scenario,
            PartitionStreamCase::StopOnPoll | PartitionStreamCase::StopWithError
        ) {
            producer_control.stopped.store(true, Ordering::Relaxed);
        }
        item
    });
    let schema = if matches!(scenario, PartitionStreamCase::StreamSchema) {
        Arc::new(Schema::empty())
    } else {
        VECTOR_RESULT_SCHEMA.clone()
    };
    let source = Box::pin(RecordBatchStreamAdapter::new(schema, source));
    let result =
        super::super::rabit_scan::controlled_partition_stream(source, 2, 1, control.clone());
    let mut received = 0;
    let mut failure = None;
    match result {
        Err(error) => {
            assert!(matches!(&error, Error::Internal { .. }));
            failure = Some(error.to_string());
        }
        Ok(mut batches) => {
            while let Some(item) = batches.next().await {
                match item {
                    Ok(_) => received += 1,
                    Err(error) => {
                        assert!(matches!(&error, DataFusionError::External(_)));
                        failure = Some(error.to_string());
                        assert!(
                            batches.next().await.is_none(),
                            "errors must terminate the stream"
                        );
                        break;
                    }
                }
            }
        }
    }
    match (failure, expected_error) {
        (Some(error), Some(message)) => assert!(error.contains(message), "{error}"),
        (None, None) => {}
        actual => panic!("unexpected stream outcome: {actual:?}"),
    }
    assert_eq!(received, expected_batches);
    assert_eq!(control.batches.load(Ordering::Relaxed), expected_batches);
    assert_eq!(control.rows.load(Ordering::Relaxed), expected_rows);
    assert!(dropped.load(Ordering::SeqCst));
}

#[rstest]
#[tokio::test]
async fn controlled_partition_stream_releases_pending_source(#[values(false, true)] stop: bool) {
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = StreamDrop(dropped.clone());
    let batch = RecordBatch::try_new(
        VECTOR_RESULT_SCHEMA.clone(),
        vec![
            Arc::new(Float32Array::from(vec![0.0])),
            Arc::new(UInt64Array::from(vec![1])),
        ],
    )
    .unwrap();
    let source = futures::stream::iter(vec![Ok::<_, DataFusionError>(batch)])
        .chain(futures::stream::pending())
        .map(move |item| {
            let _keep_alive = &guard;
            item
        });
    let source = Box::pin(RecordBatchStreamAdapter::new(
        VECTOR_RESULT_SCHEMA.clone(),
        source,
    ));
    let control = Arc::new(RowBudget {
        limit: if stop { 1 } else { usize::MAX },
        ..Default::default()
    });
    let mut output =
        super::super::rabit_scan::controlled_partition_stream(source, 2, 1, control.clone())
            .unwrap();
    assert_eq!(output.next().await.unwrap().unwrap().num_rows(), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), stop);
    if stop {
        assert!(output.next().await.is_none());
    } else {
        assert!(output.next().now_or_never().is_none());
    }
    drop(output);
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(control.batches.load(Ordering::Relaxed), 1);
}

#[rstest]
#[case::compaction(false)]
#[case::stable_partition(true)]
#[tokio::test]
async fn test_survivor_scoring_through_fragment_reuse(
    #[case] tagged: bool,
    #[values(ExIndex::Identity, ExIndex::Compact)] index: ExIndex,
    #[values(false, true)] filtered: bool,
    #[values(ApproxMode::Normal, ApproxMode::Accurate)] mode: ApproxMode,
) {
    let dir = TempStrDir::default();
    let session = Arc::new(
        Session::default().with_rabit_scan_factory(Arc::new(SurvivorScanFactory { index })),
    );
    let (mut dataset, batch) = native_fixture(dir.as_str()).await;
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    // Search for a deleted row: admitting its stale entry into the heap would
    // both return the wrong address and tighten the pruning threshold.
    let key = batch["vector"].as_fixed_size_list().value(7);
    dataset.delete("i % 17 = 7").await.unwrap();
    if tagged {
        let old_fragments = dataset.fragments().clone();
        let source_ids = old_fragments.iter().map(|f| f.id).collect::<Vec<_>>();
        let (transition, new_fragments) =
            crate::index::frag_reuse_reader::tests::prepare_partition(&dataset, &source_ids, 10)
                .await;
        let fri = crate::index::frag_reuse::frag_reuse_entry_appending(&dataset, vec![transition])
            .await
            .unwrap();
        let transaction = Transaction::new(
            dataset.version().version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: old_fragments.as_ref().clone(),
                    new_fragments,
                }],
                rewritten_indices: vec![],
                frag_reuse_index: Some(fri),
            },
            None,
        );
        dataset = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
    } else {
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 100,
                max_rows_per_group: 16,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    }
    assert_eq!(dataset.fragments().len(), 2);

    // A second move exercises composed translation, including a deletion
    // recorded after the original index and first mapping were written.
    dataset.delete("i = 8").await.unwrap();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), 1);
    let predicate = if filtered { "i % 3 = 0" } else { "true" };
    let live = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let live_ids = live[ROW_ID].as_primitive::<UInt64Type>();
    let filter: Arc<dyn PreFilter> = if filtered {
        let mask = Arc::new(RowAddrMask::AllowList(
            live_ids.values().iter().copied().collect(),
        ));
        Arc::new(DatasetPreFilter::new_with_filter_future(
            Arc::new(dataset.clone()),
            &dataset.load_indices().await.unwrap(),
            Some(async move { Ok(mask) }.boxed()),
        ))
    } else {
        Arc::new(NoFilter)
    };
    let ordinary = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    dataset.session = session;
    let lazy = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    for k in [8, 256] {
        let mut query = lazy_test_query(key.clone(), k, 4);
        query.approx_mode = mode;
        let expected = search_global(&ordinary, &query, filter.clone())
            .await
            .unwrap();
        let actual = search_global(&lazy, &query, filter.clone()).await.unwrap();
        assert_eq!(result_bits(&actual), result_bits(&expected));
        assert_eq!(actual.num_rows(), k.min(live.num_rows()));
        assert!(
            result_bits(&actual)
                .0
                .iter()
                .all(|row| live_ids.values().contains(row))
        );
        if k == 8 {
            let truth = dataset
                .scan()
                .filter(predicate)
                .unwrap()
                .nearest("vector", key.as_ref(), k)
                .unwrap()
                .prefilter(true)
                .use_index(false)
                .with_row_id()
                .try_into_batch()
                .await
                .unwrap();
            let truth = truth[ROW_ID].as_primitive::<UInt64Type>();
            assert_eq!(truth.len(), k);
            assert!(
                result_bits(&actual)
                    .0
                    .iter()
                    .filter(|id| truth.values().contains(id))
                    .count()
                    >= 4
            );
        }
    }
}

#[derive(Debug, Clone)]
struct SurvivorScanFactory {
    index: ExIndex,
}

impl super::super::rabit_scan::RabitScanFactory for SurvivorScanFactory {
    fn create(
        &self,
        _file: &IndexFileKey,
        _origin: OriginLatencyClass,
    ) -> Result<Arc<dyn super::super::rabit_scan::RabitScan>> {
        Ok(Arc::new(self.clone()))
    }
}

#[async_trait]
impl super::super::rabit_scan::RabitScan for SurvivorScanFactory {
    async fn search(
        &self,
        context: Arc<super::super::rabit_scan::RabitScanContext>,
    ) -> Result<Option<RecordBatch>> {
        context.pre_filter.wait_for_ready().await?;
        let mut heap: std::collections::BinaryHeap<lance_index::vector::graph::OrderedNode<u64>> =
            std::collections::BinaryHeap::new();
        for probe in context.probes.clone() {
            let partition = context.partitions.value(probe) as usize;
            let sign = context
                .prepare_sign_partition(partition, context.centroid_distances.value(probe))
                .await?
                .expect("fixture supports native sign preparation");
            let mut survivors = Vec::new();
            let mut counts = StagePruneCounts::default();
            select_full_survivors(
                &sign.stage.lower_bounds,
                sign.accept.as_deref(),
                context.query.upper_bound,
                heap.peek()
                    .filter(|_| {
                        heap.len()
                            >= context.query.k * context.query.refine_factor.unwrap_or(1) as usize
                    })
                    .map(|node| node.dist.0),
                &mut survivors,
                &mut counts,
            );
            let gathered = match self.index {
                ExIndex::Identity => GatheredRabitEx::Native {
                    batch: context
                        .storage()
                        .load_plane(partition, 1, context.cache(), context.metrics.io_stats())
                        .await?,
                    index: self.index,
                    source: None,
                    origin_row_reads: 0,
                },
                ExIndex::Compact => {
                    context
                        .storage()
                        .gather_rabit_ex_rows(
                            partition,
                            &survivors,
                            GatherPlan {
                                planes: [PlaneSource::Sparse; 2],
                                origin_whole: [false; 2],
                            },
                            None,
                            context.cache(),
                            context.metrics.io_stats(),
                        )
                        .await?
                }
            };
            context.score_rabit_survivors(
                &sign,
                &survivors,
                counts,
                &gathered,
                &mut heap,
                |_| {},
            )?;
        }
        Ok(Some(context.finish(heap)?))
    }
}

#[rstest]
#[case::unrefined(None, 8)]
#[case::unit_refinement(Some(1), 8)]
#[case::oversampled(Some(3), 24)]
#[tokio::test]
async fn test_survivor_scoring_preserves_refinement_candidates(
    #[case] refine_factor: Option<u32>,
    #[case] candidate_count: usize,
    #[values(ExIndex::Identity, ExIndex::Compact)] index: ExIndex,
) {
    let dir = TempStrDir::default();
    let (dataset, batch) = native_fixture(dir.as_str()).await;
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let mut query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 8, 4);
    query.refine_factor = refine_factor;
    let ordinary = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let expected = search_global(&ordinary, &query, Arc::new(NoFilter))
        .await
        .unwrap();
    assert_eq!(expected.num_rows(), candidate_count);
    let configured = open_with_session(
        dir.as_str(),
        Session::default().with_rabit_scan_factory(Arc::new(SurvivorScanFactory { index })),
    )
    .await;
    let strategy = configured
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let actual = search_global(&strategy, &query, Arc::new(NoFilter))
        .await
        .unwrap();
    assert_eq!(actual.num_rows(), candidate_count);
    assert_eq!(result_bits(&actual), result_bits(&expected));
    let truth =
        super::ground_truth(&dataset, "vector", query.key.as_ref(), 8, DistanceType::L2).await;
    assert!(
        result_bits(&actual)
            .0
            .iter()
            .filter(|row| truth.contains(row))
            .count()
            >= 4,
        "recall must be >= 0.5"
    );
}
const LAZY_METADATA_CACHE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone)]
struct RecordingScanFactory {
    opens: Arc<AtomicUsize>,
    searches: Arc<AtomicUsize>,
    behavior: TestScanBehavior,
    cache_layout: RabitCacheLayout,
}

#[derive(Debug, Clone, Copy)]
enum TestScanBehavior {
    Full,
    SearchError,
    OpenError,
}

impl super::super::rabit_scan::RabitScanFactory for RecordingScanFactory {
    fn create(
        &self,
        _file: &IndexFileKey,
        _origin: OriginLatencyClass,
    ) -> Result<Arc<dyn super::super::rabit_scan::RabitScan>> {
        self.opens.fetch_add(1, Ordering::Relaxed);
        if matches!(self.behavior, TestScanBehavior::OpenError) {
            return Err(Error::invalid_input("injected RaBitQ factory failure"));
        }
        Ok(Arc::new(self.clone()))
    }
}

#[async_trait]
impl super::super::rabit_scan::RabitScan for RecordingScanFactory {
    fn cache_layout(&self) -> RabitCacheLayout {
        self.cache_layout
    }

    async fn search(
        &self,
        _context: Arc<super::super::rabit_scan::RabitScanContext>,
    ) -> Result<Option<RecordBatch>> {
        self.searches.fetch_add(1, Ordering::Relaxed);
        if matches!(self.behavior, TestScanBehavior::SearchError) {
            Err(Error::internal("injected RaBitQ execution failure"))
        } else {
            Ok(None)
        }
    }
}

#[rstest]
#[case::fallback(TestScanBehavior::Full)]
#[case::execution_error(TestScanBehavior::SearchError)]
#[case::open_error(TestScanBehavior::OpenError)]
#[tokio::test]
async fn test_rabit_scan_factory_rebinds_from_current_session(
    #[case] behavior: TestScanBehavior,
    #[values(RabitCacheLayout::Partition, RabitCacheLayout::Planes)] cache_layout: RabitCacheLayout,
) {
    let dir = TempStrDir::default();
    let (dataset, batch) = native_fixture(dir.as_str()).await;
    let vectors = batch["vector"].as_fixed_size_list();
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let query = lazy_test_query(vectors.value(0), 8, 4);
    let truth =
        super::ground_truth(&dataset, "vector", query.key.as_ref(), 8, DistanceType::L2).await;
    let backend = Arc::new(QuickCacheBackend::with_capacity(4 * 1024 * 1024));
    let session = || {
        Session::with_index_cache_backend(
            backend.clone(),
            LAZY_METADATA_CACHE_BYTES,
            Arc::default(),
        )
    };
    let original = Arc::new(RecordingScanFactory {
        opens: Arc::default(),
        searches: Arc::default(),
        behavior: TestScanBehavior::Full,
        cache_layout,
    });
    let installed = open_with_session(
        dir.as_str(),
        session().with_rabit_scan_factory(original.clone()),
    )
    .await;
    let mut expected = None;
    for opening in 1..=2 {
        let index = installed
            .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            lazy_index(&index).uses_rabit_plane_cache(),
            cache_layout == RabitCacheLayout::Planes
        );
        let result = search_global(&index, &query, Arc::new(NoFilter))
            .await
            .unwrap();
        let actual = result_bits(&result);
        assert_eq!(original.opens.load(Ordering::Relaxed), opening);
        assert_eq!(original.searches.load(Ordering::Relaxed), opening);
        assert!(
            actual.0.iter().filter(|row| truth.contains(row)).count() >= 4,
            "recall must be >= 0.5"
        );
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected);
        }
        expected = Some(actual);
    }
    let expected = expected.unwrap();

    // Sharing serialized file state never installs another session's strategy.
    let ordinary = open_with_session(dir.as_str(), session()).await;
    let index = ordinary
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(lazy_index(&index).rabit_scan.is_none());
    assert!(!lazy_index(&index).uses_rabit_plane_cache());
    assert_eq!(
        result_bits(
            &search_global(&index, &query, Arc::new(NoFilter))
                .await
                .unwrap()
        ),
        expected
    );
    assert_eq!(original.searches.load(Ordering::Relaxed), 2);

    let replacement_layout = match cache_layout {
        RabitCacheLayout::Partition => RabitCacheLayout::Planes,
        RabitCacheLayout::Planes => RabitCacheLayout::Partition,
    };
    let replacement = Arc::new(RecordingScanFactory {
        behavior,
        cache_layout: replacement_layout,
        opens: Arc::default(),
        searches: Arc::default(),
    });
    let replaced = open_with_session(
        dir.as_str(),
        session().with_rabit_scan_factory(replacement.clone()),
    )
    .await;
    let opened = replaced
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await;
    assert_eq!(replacement.opens.load(Ordering::Relaxed), 1);
    if matches!(behavior, TestScanBehavior::OpenError) {
        let error = opened.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("injected RaBitQ factory failure")
        );
        return;
    }
    let opened = opened.unwrap();
    assert_eq!(
        lazy_index(&opened).uses_rabit_plane_cache(),
        replacement_layout == RabitCacheLayout::Planes
    );
    let result = search_global(&opened, &query, Arc::new(NoFilter)).await;
    assert_eq!(replacement.searches.load(Ordering::Relaxed), 1);
    match behavior {
        TestScanBehavior::Full => assert_eq!(result_bits(&result.unwrap()), expected),
        TestScanBehavior::SearchError => {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::Internal { .. }), "{error}");
            assert!(
                error
                    .to_string()
                    .contains("injected RaBitQ execution failure")
            );
        }
        TestScanBehavior::OpenError => unreachable!(),
    }
}
