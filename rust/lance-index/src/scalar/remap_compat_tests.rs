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
