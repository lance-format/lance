// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The remap compatibility contract: an index that only implements the
//! legacy in-memory `remap` keeps working through `remap_streaming` (a
//! synchronous translator is forwarded, a batch translator is materialized
//! first), and the built-in indices never enter that fallback.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lance_core::cache::LanceCache;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::utils::tempfile::TempObjDir;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use roaring::RoaringBitmap;

use crate::metrics::MetricsCollector;
use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::{
    AnyQuery, BatchRowIdRemapper, CreatedIndex, IndexStore, RemapUnavailable, RowAddrTranslator,
    ScalarIndex, ScalarIndexParams, SearchResult, UpdateCriteria,
};
use crate::{Index, IndexType};

/// A plugin index written against the legacy API: `remap` only, plus the
/// list of fragments its files hold.
#[derive(Debug, DeepSizeOf)]
struct LegacyOnlyIndex {
    stored: Option<Vec<u32>>,
    /// `(pointer of the map handed to remap, snapshot of it)` per call.
    remaps: Mutex<Vec<(usize, RowAddrRemap)>>,
}

impl LegacyOnlyIndex {
    fn new(stored: Option<Vec<u32>>) -> Self {
        Self {
            stored,
            remaps: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Index for LegacyOnlyIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }
    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::Value::Null)
    }
    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }
    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }
    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        Ok(RoaringBitmap::new())
    }
}

#[async_trait]
impl ScalarIndex for LegacyOnlyIndex {
    async fn search(&self, _: &dyn AnyQuery, _: &dyn MetricsCollector) -> Result<SearchResult> {
        unimplemented!()
    }
    fn can_remap(&self) -> bool {
        true
    }
    async fn remap(&self, mapping: &RowAddrRemap, _: &dyn IndexStore) -> Result<CreatedIndex> {
        self.remaps
            .lock()
            .unwrap()
            .push((mapping as *const RowAddrRemap as usize, mapping.clone()));
        Ok(CreatedIndex {
            index_details: prost_types::Any::default(),
            index_version: 0,
            files: vec![],
        })
    }
    fn stored_fragments(&self) -> Option<RoaringBitmap> {
        self.stored
            .as_ref()
            .map(|fragments| fragments.iter().copied().collect())
    }
    async fn update(
        &self,
        _: datafusion::execution::SendableRecordBatchStream,
        _: &dyn IndexStore,
        _: Option<crate::scalar::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        unimplemented!()
    }
    fn update_criteria(&self) -> UpdateCriteria {
        unimplemented!()
    }
    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        unimplemented!()
    }
}

/// Fragment 1 (4 rows, offset 1 deleted, the rest move to fragment 5),
/// fragment 3 (2 rows, excluded). Counts every question asked of it.
#[derive(Debug)]
struct Hops {
    translations: AtomicUsize,
    sizings: AtomicUsize,
    budget: u64,
}

#[async_trait]
impl BatchRowIdRemapper for Hops {
    async fn remap_row_ids(&self, ids: &[u64]) -> Result<Vec<Option<u64>>> {
        self.translations.fetch_add(1, Ordering::Relaxed);
        Ok(ids
            .iter()
            .map(|&address| {
                let fragment = RowAddress::from(address).fragment_id();
                let offset = u64::from(RowAddress::from(address).row_offset());
                match fragment {
                    1 if offset == 1 => None,
                    1 => Some(u64::from(RowAddress::new_from_parts(5, offset as u32))),
                    3 => None,
                    _ => Some(address),
                }
            })
            .collect())
    }
    fn fragment_physical_rows(&self, fragment: u32) -> Option<u64> {
        self.sizings.fetch_add(1, Ordering::Relaxed);
        match fragment {
            1 => Some(4),
            3 => Some(2),
            _ => None,
        }
    }
    fn materialization_budget_bytes(&self) -> u64 {
        self.budget
    }
}

fn hops_with_budget(budget: u64) -> Arc<Hops> {
    Arc::new(Hops {
        translations: AtomicUsize::new(0),
        sizings: AtomicUsize::new(0),
        budget,
    })
}

fn temp_store() -> (TempObjDir, Arc<LanceIndexStore>) {
    let dir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        dir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    (dir, store)
}

#[tokio::test]
async fn legacy_only_index_gets_a_synchronous_map_as_it_is() {
    let (_dir, store) = temp_store();
    let index = LegacyOnlyIndex::new(None);
    let map = Arc::new(RowAddrRemap::direct(
        [(RowAddress::new_from_parts(1, 0).into(), None)].into(),
    ));
    let translator = RowAddrTranslator::Sync(map.clone());
    index
        .remap_streaming(&translator, store.as_ref())
        .await
        .unwrap();
    let remaps = index.remaps.lock().unwrap();
    assert_eq!(remaps.len(), 1, "one call to the legacy remap");
    assert_eq!(
        remaps[0].0,
        Arc::as_ptr(&map) as usize,
        "the very same map, not a copy"
    );
}

#[tokio::test]
async fn legacy_only_index_gets_a_complete_materialized_map_once() {
    let (_dir, store) = temp_store();
    let index = LegacyOnlyIndex::new(Some(vec![1, 3]));
    let hops = hops_with_budget(u64::MAX);
    let translator = RowAddrTranslator::Batch(hops.clone());
    index
        .remap_streaming(&translator, store.as_ref())
        .await
        .unwrap();
    let remaps = index.remaps.lock().unwrap();
    assert_eq!(remaps.len(), 1, "one call to the legacy remap");
    let map = &remaps[0].1;
    let at = |fragment: u32, offset: u32| u64::from(RowAddress::new_from_parts(fragment, offset));
    assert_eq!(map.get(at(1, 0)), Some(Some(at(5, 0))));
    assert_eq!(map.get(at(1, 1)), Some(None), "the deleted row is explicit");
    assert_eq!(map.get(at(1, 3)), Some(Some(at(5, 3))));
    // The excluded fragment is mapped to None row by row: the legacy remap
    // would otherwise keep its stale addresses.
    assert_eq!(map.get(at(3, 0)), Some(None));
    assert_eq!(map.get(at(3, 1)), Some(None));
    assert_eq!(
        map.get(at(3, 2)),
        None,
        "nothing beyond the fragment's rows"
    );
    assert!(hops.sizings.load(Ordering::Relaxed) >= 2);
    assert_eq!(
        hops.translations.load(Ordering::Relaxed),
        2,
        "one slice per fragment"
    );
}

#[tokio::test]
async fn legacy_only_index_is_left_alone_when_the_fallback_cannot_be_proven() {
    let (_dir, store) = temp_store();
    // Unknown stored fragments.
    let index = LegacyOnlyIndex::new(None);
    let hops = hops_with_budget(u64::MAX);
    let Err(error) = index
        .remap_streaming(&RowAddrTranslator::Batch(hops.clone()), store.as_ref())
        .await
    else {
        panic!("the fallback cannot run without stored fragments");
    };
    assert_eq!(
        RemapUnavailable::from_error(&error),
        Some(&RemapUnavailable::StoredFragmentsUnknown)
    );
    assert!(
        index.remaps.lock().unwrap().is_empty(),
        "the legacy remap never ran"
    );
    assert_eq!(hops.translations.load(Ordering::Relaxed), 0);
    // Over budget.
    let index = LegacyOnlyIndex::new(Some(vec![1, 3]));
    let hops = hops_with_budget(16);
    let Err(error) = index
        .remap_streaming(&RowAddrTranslator::Batch(hops.clone()), store.as_ref())
        .await
    else {
        panic!("the fallback cannot run over budget");
    };
    assert!(matches!(
        RemapUnavailable::from_error(&error),
        Some(RemapUnavailable::OverBudget { .. })
    ));
    assert!(index.remaps.lock().unwrap().is_empty());
    assert_eq!(hops.translations.load(Ordering::Relaxed), 0);
    // A plain error is not a skip.
    let error = Error::io("disk");
    assert!(RemapUnavailable::from_error(&error).is_none());
}

// The built-in plugins that rebuild their segment from stored row ids
// (MinHash LSH) or wrap another plugin (JSON): `remap_streaming` under a
// batch translator writes what the legacy `remap` writes for the same
// mapping, without materializing the map, and drops the deleted and
// excluded rows while leaving unmapped addresses alone.

/// The addresses the `Hops` fixture is about, one indexed row each:
/// fragment 1 (offset 1 deleted, the rest moved to fragment 5), fragment 3
/// (excluded) and fragment 2 (untouched).
fn hops_addresses() -> Vec<u64> {
    [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (3, 0),
        (3, 1),
        (2, 0),
        (2, 1),
    ]
    .into_iter()
    .map(|(fragment, offset)| u64::from(RowAddress::new_from_parts(fragment, offset)))
    .collect()
}

/// What `Hops` does, as the in-memory map the legacy `remap` takes.
fn hops_as_legacy_map() -> RowAddrRemap {
    let at = |fragment: u32, offset: u32| u64::from(RowAddress::new_from_parts(fragment, offset));
    RowAddrRemap::direct(
        [
            (at(1, 0), Some(at(5, 0))),
            (at(1, 1), None),
            (at(1, 2), Some(at(5, 2))),
            (at(1, 3), Some(at(5, 3))),
            (at(3, 0), None),
            (at(3, 1), None),
        ]
        .into(),
    )
}

/// The addresses a remap through `Hops` leaves in the segment.
fn hops_expected_addresses() -> Vec<u64> {
    let mut expected: Vec<u64> = [(5, 0), (5, 2), (5, 3), (2, 0), (2, 1)]
        .into_iter()
        .map(|(fragment, offset)| u64::from(RowAddress::new_from_parts(fragment, offset)))
        .collect();
    expected.sort_unstable();
    expected
}

fn one_batch_stream(
    schema: Arc<arrow_schema::Schema>,
    columns: Vec<arrow_array::ArrayRef>,
) -> datafusion::execution::SendableRecordBatchStream {
    let batch = arrow_array::RecordBatch::try_new(schema.clone(), columns).unwrap();
    Box::pin(
        datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(vec![Ok(batch)]),
        ),
    )
}

/// Train an index of `plugin` over `(value, row id)` rows through the
/// plugin's trainer, as a dataset would.
async fn train(
    plugin: &str,
    params: &str,
    value_field: arrow_schema::Field,
    values: arrow_array::ArrayRef,
    row_ids: &[u64],
    store: &dyn IndexStore,
) -> CreatedIndex {
    use crate::registry::IndexPluginRegistry;
    use crate::scalar::registry::VALUE_COLUMN_NAME;
    let registry = IndexPluginRegistry::with_default_plugins();
    let plugin = registry.get_plugin_by_name(plugin).unwrap();
    let trainer = plugin.basic_trainer().unwrap();
    let request = trainer
        .new_training_request(
            params,
            &arrow_schema::Field::new(VALUE_COLUMN_NAME, value_field.data_type().clone(), true),
        )
        .unwrap();
    assert!(request.criteria().needs_row_ids);
    assert!(!request.criteria().needs_row_addrs);
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        value_field.with_name(VALUE_COLUMN_NAME),
        arrow_schema::Field::new(lance_core::ROW_ID, arrow_schema::DataType::UInt64, false),
    ]));
    let data = one_batch_stream(
        schema,
        vec![
            values,
            Arc::new(arrow_array::UInt64Array::from(row_ids.to_vec())),
        ],
    );
    trainer
        .train_index(data, store, request, None, crate::progress::noop_progress())
        .await
        .unwrap()
}

/// Remap `index` both ways into fresh stores: through the legacy in-memory
/// map and through `remap_streaming` with the `Hops` batch translator, which
/// must translate (not size) and never enter the materializing fallback.
async fn remap_both_ways(
    index: &dyn ScalarIndex,
) -> (
    (TempObjDir, Arc<LanceIndexStore>),
    (TempObjDir, Arc<LanceIndexStore>),
) {
    let legacy = temp_store();
    index
        .remap(&hops_as_legacy_map(), legacy.1.as_ref())
        .await
        .unwrap();
    let streamed = temp_store();
    let hops = hops_with_budget(0);
    index
        .remap_streaming(&RowAddrTranslator::Batch(hops.clone()), streamed.1.as_ref())
        .await
        .unwrap();
    assert_eq!(hops.sizings.load(Ordering::Relaxed), 0, "no fallback");
    assert!(hops.translations.load(Ordering::Relaxed) > 0);
    (legacy, streamed)
}

const MINHASH_PARAMS: &str = r#"{"num_hashes": 32, "num_bands": 8, "shingle_size": 1}"#;

const MINHASH_TEXTS: [&str; 8] = [
    "apple banana cherry",
    "mango sorbet bowl",
    "pear tart slice",
    "lemon curd jar",
    "peach cobbler dish",
    "plum pudding cup",
    "fig newton bar",
    "kiwi lime soda",
];

/// The row ids of a MinHash LSH segment's signature table, sorted.
async fn minhash_stored_row_ids(store: &dyn IndexStore) -> Vec<u64> {
    use crate::scalar::minhash_lsh::SIGNATURES_FILENAME;
    use arrow_array::cast::AsArray;
    let reader = store.open_index_file(SIGNATURES_FILENAME).await.unwrap();
    let batch = reader
        .read_range(0..reader.num_rows(), Some(&[lance_core::ROW_ID]))
        .await
        .unwrap();
    let mut row_ids: Vec<u64> = batch[lance_core::ROW_ID]
        .as_primitive::<arrow_array::types::UInt64Type>()
        .values()
        .to_vec();
    row_ids.sort_unstable();
    row_ids
}

#[tokio::test]
async fn minhash_remap_streaming_matches_legacy_remap_without_the_fallback() {
    use crate::metrics::NoOpMetricsCollector;
    use crate::scalar::minhash_lsh::MinHashLshIndex;
    use lance_select::RowAddrMask;

    let (_src_dir, src_store) = temp_store();
    let created = train(
        "minhashlsh",
        MINHASH_PARAMS,
        arrow_schema::Field::new("text", arrow_schema::DataType::Utf8, true),
        Arc::new(arrow_array::StringArray::from(MINHASH_TEXTS.to_vec())),
        &hops_addresses(),
        src_store.as_ref(),
    )
    .await;
    let cache = LanceCache::no_cache();
    let index = MinHashLshIndex::load(src_store.clone(), &created.index_details, None, &cache)
        .await
        .unwrap();
    assert_eq!(minhash_stored_row_ids(src_store.as_ref()).await, {
        let mut all = hops_addresses();
        all.sort_unstable();
        all
    });

    let (legacy, streamed) = remap_both_ways(index.as_ref()).await;
    let streamed_rows = minhash_stored_row_ids(streamed.1.as_ref()).await;
    assert_eq!(
        streamed_rows,
        minhash_stored_row_ids(legacy.1.as_ref()).await
    );
    assert_eq!(streamed_rows, hops_expected_addresses());

    // The rewritten segment answers by the new addresses: the moved row by
    // its destination, the untouched row by its own address, and the
    // deleted row's text finds no identical signature anymore.
    let remapped = MinHashLshIndex::load(streamed.1.clone(), &created.index_details, None, &cache)
        .await
        .unwrap();
    let top = |text: &'static str| async {
        remapped
            .search_text(text, 1, &RowAddrMask::all_rows(), &NoOpMetricsCollector)
            .await
            .unwrap()
    };
    let at = |fragment: u32, offset: u32| u64::from(RowAddress::new_from_parts(fragment, offset));
    assert_eq!(top(MINHASH_TEXTS[0]).await[0].row_id, at(5, 0));
    assert_eq!(top(MINHASH_TEXTS[6]).await[0].row_id, at(2, 0));
    assert!(
        top(MINHASH_TEXTS[1]).await.is_empty(),
        "the deleted row is gone"
    );
}

/// A segment opened with the batch remapper of a tagged history translates
/// its stored row ids as it scores candidates: a moved row is reported at
/// its destination, a deleted row is dropped, an untouched row is its own.
#[tokio::test]
async fn minhash_query_translates_candidates_through_the_batch_remapper() {
    use crate::metrics::NoOpMetricsCollector;
    use crate::scalar::minhash_lsh::MinHashLshIndex;
    use lance_select::RowAddrMask;

    let (_src_dir, src_store) = temp_store();
    let created = train(
        "minhashlsh",
        MINHASH_PARAMS,
        arrow_schema::Field::new("text", arrow_schema::DataType::Utf8, true),
        Arc::new(arrow_array::StringArray::from(MINHASH_TEXTS.to_vec())),
        &hops_addresses(),
        src_store.as_ref(),
    )
    .await;
    let cache = LanceCache::no_cache();
    let hops = hops_with_budget(0);
    let index = MinHashLshIndex::load_with_remapping(
        src_store.clone(),
        &created.index_details,
        Some(hops.clone()),
        &cache,
    )
    .await
    .unwrap();
    let top = |text: &'static str| async {
        index
            .search_text(text, 1, &RowAddrMask::all_rows(), &NoOpMetricsCollector)
            .await
            .unwrap()
    };
    let at = |fragment: u32, offset: u32| u64::from(RowAddress::new_from_parts(fragment, offset));
    assert_eq!(top(MINHASH_TEXTS[0]).await[0].row_id, at(5, 0));
    assert_eq!(top(MINHASH_TEXTS[3]).await[0].row_id, at(5, 3));
    assert_eq!(top(MINHASH_TEXTS[6]).await[0].row_id, at(2, 0));
    assert!(top(MINHASH_TEXTS[1]).await.is_empty(), "the deleted row");
    assert!(top(MINHASH_TEXTS[4]).await.is_empty(), "an excluded row");
    assert!(hops.translations.load(Ordering::Relaxed) > 0);
    assert_eq!(hops.sizings.load(Ordering::Relaxed), 0);

    // The same segment rebuilt from that opening carries the translated
    // addresses (and nothing for the dropped rows) into the new segment.
    let (_dest_dir, dest_store) = temp_store();
    index
        .remap_streaming(
            &RowAddrTranslator::Batch(hops_with_budget(0)),
            dest_store.as_ref(),
        )
        .await
        .unwrap();
    assert_eq!(
        minhash_stored_row_ids(dest_store.as_ref()).await,
        hops_expected_addresses()
    );
}

/// The JSON-path wrapper over a BTree: a row address per value, so the
/// rewritten target can be read back value by value.
async fn json_addresses_by_value(
    store: Arc<LanceIndexStore>,
    details: &prost_types::Any,
) -> Vec<(i64, Vec<u64>)> {
    use crate::metrics::NoOpMetricsCollector;
    use crate::registry::IndexPluginRegistry;
    use crate::scalar::json::JsonQuery;
    use crate::scalar::{SargableQuery, SearchResult};
    use datafusion_common::ScalarValue;
    let registry = IndexPluginRegistry::with_default_plugins();
    let plugin = registry.get_plugin_by_name("json").unwrap();
    let index = plugin
        .load_index(store, details, 0, None, &LanceCache::no_cache())
        .await
        .unwrap();
    let mut found = Vec::new();
    for value in 0..8i64 {
        let query = JsonQuery::new(
            Arc::new(SargableQuery::Equals(ScalarValue::Int64(Some(value)))),
            "v".to_string(),
        );
        let SearchResult::Exact(rows) = index.search(&query, &NoOpMetricsCollector).await.unwrap()
        else {
            panic!("expected an exact result");
        };
        let addrs: Vec<u64> = rows
            .true_rows()
            .row_addrs()
            .unwrap()
            .map(u64::from)
            .collect();
        found.push((value, addrs));
    }
    found
}

#[tokio::test]
async fn json_remap_streaming_delegates_to_the_target_without_the_fallback() {
    use crate::registry::IndexPluginRegistry;

    let (_src_dir, src_store) = temp_store();
    let docs: Vec<Vec<u8>> = (0..8)
        .map(|value| {
            format!(r#"{{"v": {value}}}"#)
                .parse::<jsonb::OwnedJsonb>()
                .unwrap()
                .to_vec()
        })
        .collect();
    let created = train(
        "json",
        r#"{"target_index_type": "btree", "path": "v"}"#,
        arrow_schema::Field::new("json", arrow_schema::DataType::LargeBinary, true),
        Arc::new(arrow_array::LargeBinaryArray::from(
            docs.iter()
                .map(|doc| Some(doc.as_slice()))
                .collect::<Vec<_>>(),
        )),
        &hops_addresses(),
        src_store.as_ref(),
    )
    .await;
    let registry = IndexPluginRegistry::with_default_plugins();
    let plugin = registry.get_plugin_by_name("json").unwrap();
    assert!(plugin.supports_batch_row_id_remapping_for(&created.index_details));
    let index = plugin
        .load_index(
            src_store.clone(),
            &created.index_details,
            0,
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();

    let (legacy, streamed) = remap_both_ways(index.as_ref()).await;
    let streamed_rows = json_addresses_by_value(streamed.1.clone(), &created.index_details).await;
    assert_eq!(
        streamed_rows,
        json_addresses_by_value(legacy.1.clone(), &created.index_details).await
    );
    let at =
        |fragment: u32, offset: u32| vec![u64::from(RowAddress::new_from_parts(fragment, offset))];
    assert_eq!(
        streamed_rows,
        vec![
            (0, at(5, 0)),
            (1, vec![]),
            (2, at(5, 2)),
            (3, at(5, 3)),
            (4, vec![]),
            (5, vec![]),
            (6, at(2, 0)),
            (7, at(2, 1)),
        ]
    );

    // Opened under the batch remapper, the wrapper answers through its
    // translated target.
    let translated = plugin
        .load_index_with_remapping(
            src_store.clone(),
            &created.index_details,
            0,
            Some(hops_with_budget(0)),
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();
    {
        use crate::metrics::NoOpMetricsCollector;
        use crate::scalar::json::JsonQuery;
        use crate::scalar::{SargableQuery, SearchResult};
        use datafusion_common::ScalarValue;
        for (value, expected) in [(0i64, at(5, 0)), (1, vec![]), (4, vec![]), (7, at(2, 1))] {
            let query = JsonQuery::new(
                Arc::new(SargableQuery::Equals(ScalarValue::Int64(Some(value)))),
                "v".to_string(),
            );
            let SearchResult::Exact(rows) = translated
                .search(&query, &NoOpMetricsCollector)
                .await
                .unwrap()
            else {
                panic!("expected an exact result");
            };
            let addrs: Vec<u64> = rows
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(u64::from)
                .collect();
            assert_eq!(addrs, expected, "v = {value}");
        }
    }
}

/// The JSON wrapper answers for its target: an FM target has no batch
/// remapper, so the wrapper reports none for that segment and refuses to
/// load it under a batch translator; without details it promises nothing.
#[tokio::test]
async fn json_batch_remapping_capability_follows_the_target() {
    use crate::registry::IndexPluginRegistry;

    let registry = IndexPluginRegistry::with_default_plugins();
    let plugin = registry.get_plugin_by_name("json").unwrap();
    assert!(!plugin.supports_batch_row_id_remapping());
    let over_fm = prost_types::Any::from_msg(&crate::pb::JsonIndexDetails {
        path: "v".to_string(),
        target_details: Some(prost_types::Any::from_msg(&crate::pb::FmIndexDetails {}).unwrap()),
    })
    .unwrap();
    assert!(!plugin.supports_batch_row_id_remapping_for(&over_fm));
    let undecodable = prost_types::Any {
        type_url: over_fm.type_url.clone(),
        value: vec![0xff],
    };
    assert!(!plugin.supports_batch_row_id_remapping_for(&undecodable));

    let (_dir, store) = temp_store();
    let err = plugin
        .load_index_with_remapping(
            store.clone(),
            &over_fm,
            0,
            Some(hops_with_budget(0)),
            &LanceCache::no_cache(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
}
