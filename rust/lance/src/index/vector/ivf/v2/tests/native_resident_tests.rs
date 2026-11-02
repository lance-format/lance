// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::IVFIndex;
use super::native_storage_tests::*;
use super::*;
use crate::session::{ResidentColumnsSetting, Session};
use lance_core::cache::QuickCacheBackend;
use lance_index::vector::bq::builder::RabitQuantizer;

pub(super) async fn open_with_session(uri: &str, session: Session) -> Dataset {
    crate::DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(session))
        .load()
        .await
        .unwrap()
}

#[tokio::test]
async fn native_resident_store_is_ready_after_open_and_reconstruction() {
    let dir = TempStrDir::default();
    let (dataset, _) = native_fixture(dir.as_str()).await;
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let backend = Arc::new(QuickCacheBackend::with_capacity(4 * 1024 * 1024));
    for _ in 0..2 {
        let session =
            Session::with_index_cache_backend(backend.clone(), 4 * 1024 * 1024, Arc::default())
                .with_index_resident_columns(ResidentColumnsSetting::On);
        let opened = open_with_session(dir.as_str(), session).await;
        let index = opened
            .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let index = index
            .as_any()
            .downcast_ref::<IVFIndex<FlatIndex, RabitQuantizer>>()
            .unwrap();
        assert!(index.resident_columns_enabled());
        assert!(index.resident_columns_bytes() > 0);
        assert_eq!(
            index.storage.resident_columns().loaded_bytes(),
            Some(index.resident_columns_bytes()),
            "fresh opens and cached-state reconstruction must finish loading before returning"
        );
    }
}
