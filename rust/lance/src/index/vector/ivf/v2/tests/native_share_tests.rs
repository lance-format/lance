// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::IVFIndex;
use super::native_resident_tests::open_with_session;
use super::native_storage_tests::native_fixture;
use super::*;
use crate::session::{ResidentColumnsSetting, Session};
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::storage::{ResidentAttach, shares_resident_store};

#[tokio::test]
async fn native_query_views_share_but_exported_partitions_own_resident_rows() {
    let dir = TempStrDir::default();
    let (dataset, _) = native_fixture(dir.as_str()).await;
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let opened = open_with_session(
        dir.as_str(),
        Session::default().with_index_resident_columns(ResidentColumnsSetting::On),
    )
    .await;
    let index = opened
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let index = index
        .as_any()
        .downcast_ref::<IVFIndex<FlatIndex, RabitQuantizer>>()
        .unwrap();
    assert_eq!(index.resident_attach(), ResidentAttach::Share);
    let part = (0..index.storage.num_partitions())
        .find(|&part| index.storage.partition_size(part) > 0)
        .unwrap();
    let query = index
        .load_partition(part, true, &NoOpMetricsCollector)
        .await
        .unwrap();
    let query_batch = query.storage.to_batches().unwrap().next().unwrap();
    assert!(shares_resident_store(&query_batch));
    let exported = index
        .partition_reader(part, true, &NoOpMetricsCollector)
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(exported.len(), 1);
    assert_eq!(exported[0], query_batch);
    assert!(!shares_resident_store(&exported[0]));
}
