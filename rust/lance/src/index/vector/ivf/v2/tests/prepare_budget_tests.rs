// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::native_storage_tests::{lazy_test_query, result_bits, search_global};
use super::*;
use lance_core::cache::{CacheCodec, CacheEntry, CacheLoader, InternalCacheKey};
use lance_index::vector::PartitionSearchControl;
use std::time::Duration;
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio::time::timeout;

/// Each cache read waits for the test to finish that specific partition load.
#[derive(Debug)]
struct PausedCache(mpsc::UnboundedSender<oneshot::Sender<()>>);

#[async_trait::async_trait]
impl CacheBackend for PausedCache {
    async fn get(&self, _: &InternalCacheKey, _: Option<CacheCodec>) -> Option<CacheEntry> {
        None
    }

    async fn insert(&self, _: &InternalCacheKey, _: CacheEntry, _: usize, _: Option<CacheCodec>) {}

    async fn get_or_insert<'a>(
        &self,
        _: &InternalCacheKey,
        loader: CacheLoader<'a>,
        _: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        let (tx, rx) = oneshot::channel();
        self.0.send(tx).unwrap();
        rx.await.unwrap();
        Ok((loader.await?.0, false))
    }

    async fn clear(&self) {}
    async fn num_entries(&self) -> usize {
        0
    }
    async fn size_bytes(&self) -> usize {
        0
    }
}

struct Fixture {
    reference: Arc<dyn VectorIndex>,
    paused: Arc<dyn VectorIndex>,
    query: Query,
    loads: mpsc::UnboundedReceiver<oneshot::Sender<()>>,
    budget: Arc<Semaphore>,
    _cache: LanceCache,
    _dataset: Dataset,
}

impl Fixture {
    async fn new(slots: usize) -> Self {
        let (batch, schema) = make_seeded_vector_batch(256);
        let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 4);
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &VectorIndexParams::ivf_flat(4, DistanceType::L2),
                true,
            )
            .await
            .unwrap();
        let uuid = dataset.load_indices().await.unwrap()[0].uuid;
        let reference = dataset
            .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let (loads_tx, loads) = mpsc::unbounded_channel();
        let cache = LanceCache::with_backend(Arc::new(PausedCache(loads_tx)));
        let state = reference
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .unwrap()
            .to_state_entry();
        let paused = state
            .0
            .reconstruct(
                dataset.table_state.object_store.clone(),
                dataset.table_state.metadata_cache.as_ref(),
                cache.clone(),
                None,
                Default::default(),
            )
            .await
            .unwrap();
        let budget = Arc::new(Semaphore::new(slots));
        for index in [&reference, &paused] {
            index
                .as_any()
                .downcast_ref::<IvfFlatIndex>()
                .unwrap()
                .test_prepare_budget
                .set(budget.clone())
                .unwrap();
        }
        Self {
            reference,
            paused,
            query,
            loads,
            budget,
            _cache: cache,
            _dataset: dataset,
        }
    }
}

#[tokio::test]
async fn direct_probes_share_admission_until_scoring() {
    let mut f = Fixture::new(2).await;
    let search = |partition| {
        f.paused.search_in_partition(
            partition,
            &f.query,
            Arc::new(NoFilter),
            &NoOpMetricsCollector,
        )
    };
    let mut probes: Vec<_> = (0..4).map(search).collect();
    for probe in &mut probes {
        assert!(futures::poll!(probe).is_pending());
    }
    let first = f.loads.try_recv().unwrap();
    let second = f.loads.try_recv().unwrap();
    assert!(f.loads.try_recv().is_err());
    assert_eq!(f.budget.available_permits(), 0);

    first.send(()).unwrap();
    let result = timeout(Duration::from_secs(3), &mut probes[0])
        .await
        .unwrap()
        .unwrap();
    for probe in &mut probes[2..] {
        assert!(futures::poll!(probe).is_pending());
    }
    let third = f.loads.try_recv().unwrap();
    assert!(f.loads.try_recv().is_err());
    assert_eq!(f.budget.available_permits(), 0);

    drop(probes);
    assert!(second.is_closed() && third.is_closed());
    let expected = f
        .reference
        .search_in_partition(0, &f.query, Arc::new(NoFilter), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(result_bits(&result), result_bits(&expected));
    assert_eq!(f.budget.available_permits(), 2);
}

struct KeepSearching;
impl PartitionSearchControl for KeepSearching {
    fn should_stop(&self) -> bool {
        false
    }
}

#[rstest]
#[case::cancel(true)]
#[case::pause(false)]
#[tokio::test]
async fn independent_search_progresses_with_stalled_reader(#[case] cancel: bool) {
    let mut f = Fixture::new(1).await;
    let expected = search_global(&f.reference, &f.query, Arc::new(NoFilter))
        .await
        .unwrap();
    // Repeat the four fixture partitions so only the channels determine when
    // preparation stops, even on a single-core runtime.
    let capacity = *super::super::STREAMING_SEARCH_BATCH_SIZE;
    let probes = capacity * 3 + 4;
    let parts = Arc::new(UInt32Array::from_iter_values(
        (0..probes).map(|i| (i % 4) as u32),
    ));
    let stream = f
        .paused
        .clone()
        .search_partitions(
            f.query.clone(),
            parts,
            Arc::new(Float32Array::from(vec![0.; probes])),
            0,
            probes,
            Arc::new(NoFilter),
            Some(Arc::new(KeepSearching)),
            Arc::new(NoOpMetricsCollector),
        )
        .await
        .unwrap();
    let first = timeout(Duration::from_secs(3), f.loads.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(f.budget.available_permits(), 0);
    let independent = search_global(&f.reference, &f.query, Arc::new(NoFilter));
    tokio::pin!(independent);
    if cancel {
        assert!(futures::poll!(&mut independent).is_pending());
        drop(stream);
        let got = timeout(Duration::from_secs(3), independent)
            .await
            .unwrap()
            .unwrap();
        assert!(first.is_closed());
        assert_eq!(result_bits(&got), result_bits(&expected));
    } else {
        first.send(()).unwrap();
        // Keep serving cache reads while the result stream stays unpolled.
        // Let the scorer fill its output and prepared-input queues before
        // starting a fresh independent request against the same semaphore.
        for _ in 1..capacity + 2 {
            timeout(Duration::from_secs(3), f.loads.recv())
                .await
                .unwrap()
                .unwrap()
                .send(())
                .unwrap();
        }
        let feed = async {
            while let Some(load) = f.loads.recv().await {
                let _ = load.send(());
            }
        };
        let got = timeout(Duration::from_secs(3), async {
            tokio::select! { got = independent => got.unwrap(), _ = feed => panic!("load source closed") }
        })
        .await
        .unwrap();
        assert_eq!(result_bits(&got), result_bits(&expected));
        drop(stream);
    }
}

#[tokio::test]
async fn batch_matches_single_queries_with_one_prepare_slot() {
    let mut f = Fixture::new(1).await;
    let second = lazy_test_query(Arc::new(Float32Array::from_value(0.5, DIM)), f.query.k, 4);
    let mut partitions = Vec::new();
    let mut distances = Vec::new();
    let mut expected = Vec::new();
    for query in [&f.query, &second] {
        let (parts, dists) = f.reference.find_partitions(query).unwrap();
        partitions.push(Arc::new(parts));
        distances.push(Arc::new(dists));
        expected.push(
            search_global(&f.reference, query, Arc::new(NoFilter))
                .await
                .unwrap(),
        );
    }
    let mut batch_query = f.query.clone();
    batch_query.key =
        arrow_select::concat::concat(&[f.query.key.as_ref(), second.key.as_ref()]).unwrap();
    let run = f.paused.clone().search_partitions_batch(
        batch_query,
        partitions,
        distances,
        Arc::new(NoFilter),
        Arc::new(NoOpMetricsCollector),
    );
    tokio::pin!(run);
    assert!(futures::poll!(&mut run).is_pending());
    let first = f.loads.try_recv().unwrap();
    assert!(f.loads.try_recv().is_err());
    assert_eq!(f.budget.available_permits(), 0);
    first.send(()).unwrap();
    let drive = async {
        while let Some(load) = f.loads.recv().await {
            load.send(()).unwrap();
        }
    };
    let batches = timeout(Duration::from_secs(3), async {
        tokio::select! { result = run => result.unwrap(), _ = drive => panic!("load source closed") }
    }).await.unwrap();
    assert_eq!(batches.len(), 2);
    // Single and batch search visit partitions in different orders and emit
    // their heaps in internal order. Compare exact ID/distance pairs per query.
    let rows = |batch: &RecordBatch| {
        let (ids, distances) = result_bits(batch);
        let mut rows = ids.into_iter().zip(distances).collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    };
    for (batch, reference) in batches.iter().zip(&expected) {
        assert_eq!(rows(batch), rows(reference));
    }
    assert_eq!(f.budget.available_permits(), 1);
}
