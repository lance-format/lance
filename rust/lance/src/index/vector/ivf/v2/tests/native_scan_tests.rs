// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::IVFIndex;
use super::native_resident_tests::open_with_session;
use super::native_storage_tests::*;
use super::*;
use crate::index::vector::ivf::v2::rabit_scan::RabitCacheLayout;
use crate::session::Session;
use async_trait::async_trait;
use lance_core::Error;
use lance_core::cache::QuickCacheBackend;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::storage::{IndexFileKey, OriginLatencyClass};

fn lazy_index(index: &Arc<dyn VectorIndex>) -> &IVFIndex<FlatIndex, RabitQuantizer> {
    index
        .as_any()
        .downcast_ref::<IVFIndex<FlatIndex, RabitQuantizer>>()
        .unwrap()
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
