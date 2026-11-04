// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The contract every scalar index that translates row addresses upholds,
//! run against each one: loaded through its
//! [`RemappingIndexLoader`](super::registry::RemappingIndexLoader) with a
//! scripted [`Remapper`], it answers with the rows' current addresses, never
//! with a stale one, and translates in batches.
//!
//! The script stands in for a fragment reuse history, so no history is
//! built here; the histories themselves are tested against their own
//! contract in lance-table.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::ScalarValue;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::cache::LanceCache;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempObjDir;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_io::object_store::ObjectStore;
use lance_select::remapper::ScriptedRemapper;
use lance_select::{RowAddrTreeMap, RowSetOps};
use rstest::rstest;

use super::btree::BTreeIndexPlugin;
use super::lance_format::LanceIndexStore;
use super::registry::{ScalarIndexPlugin, TrainingOrdering, VALUE_COLUMN_NAME};
use super::{Remapper, SargableQuery, ScalarIndex};
use crate::metrics::NoOpMetricsCollector;
use crate::progress::noop_progress;

fn address(fragment: u32, offset: u32) -> u64 {
    RowAddress::new_from_parts(fragment, offset).into()
}

/// Rows the index is built over: value `v` at `address(0, v)` for `v` in
/// `0..8`, the fragment the script rewrites, and value `100 + v` at
/// `address(2, v)` for `v` in `0..4`, a fragment it leaves alone.
fn rows() -> Vec<(i32, u64)> {
    let rewritten = (0..8).map(|value| (value, address(0, value as u32)));
    let untouched = (0..4).map(|value| (100 + value, address(2, value as u32)));
    rewritten.chain(untouched).collect()
}

/// Fragment 0 was rewritten into fragment 1 in reverse order, and the rows
/// holding values 3 and 5 were deleted.
fn script() -> ScriptedRemapper {
    (0..8u32).fold(ScriptedRemapper::new(), |script, offset| {
        if offset == 3 || offset == 5 {
            script.deleted(address(0, offset))
        } else {
            script.moved(address(0, offset), address(1, 7 - offset))
        }
    })
}

/// Where the row holding `value` is after the script.
fn current_address(value: i32) -> Option<u64> {
    match value {
        3 | 5 => None,
        0..8 => Some(address(1, 7 - value as u32)),
        _ => Some(address(2, (value - 100) as u32)),
    }
}

/// One index kind under test.
struct Case {
    plugin: Arc<dyn ScalarIndexPlugin>,
    params: &'static str,
}

fn btree() -> Case {
    Case {
        plugin: Arc::new(BTreeIndexPlugin),
        params: "{}",
    }
}

/// An index of `case`'s kind over [`rows`], loaded through its remapping
/// loader with `remapper`.
async fn load(
    case: &Case,
    store_dir: &TempObjDir,
    remapper: Option<Remapper>,
) -> Arc<dyn ScalarIndex> {
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        (**store_dir).clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let trainer = case.plugin.basic_trainer().expect("trains from a stream");
    let request = trainer
        .new_training_request(
            case.params,
            &Field::new(VALUE_COLUMN_NAME, DataType::Int32, true),
        )
        .unwrap();
    let criteria = request.criteria().clone();
    let mut rows = rows();
    match criteria.ordering {
        TrainingOrdering::Values => rows.sort_by_key(|&(value, _)| value),
        TrainingOrdering::Addresses => rows.sort_by_key(|&(_, addr)| addr),
        TrainingOrdering::None => {}
    }
    let (values, addrs): (Vec<i32>, Vec<u64>) = rows.into_iter().unzip();
    let mut fields = vec![Field::new(VALUE_COLUMN_NAME, DataType::Int32, true)];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(values))];
    // The rows have no stable IDs: their addresses stand in for them.
    for (needed, name) in [
        (criteria.needs_row_ids, ROW_ID),
        (criteria.needs_row_addrs, ROW_ADDR),
    ] {
        if needed {
            fields.push(Field::new(name, DataType::UInt64, false));
            columns.push(Arc::new(UInt64Array::from(addrs.clone())));
        }
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let data = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::once(async move { Ok(batch) }),
    ));
    let created = trainer
        .train_index(
            data,
            store.as_ref(),
            request,
            Some(vec![0, 2]),
            noop_progress(),
        )
        .await
        .unwrap();
    case.plugin
        .remapping_loader()
        .expect("translates row addresses")
        .load_remapped(
            store,
            &created.index_details,
            created.index_version,
            remapper,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap()
}

async fn search(index: &dyn ScalarIndex, value: i32) -> (RowAddrTreeMap, bool) {
    let result = index
        .search(
            &SargableQuery::Equals(ScalarValue::Int32(Some(value))),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
    (result.row_addrs().true_rows(), result.is_exact())
}

/// Each value's rows come back at their current addresses: moved rows at
/// their destination, deleted rows not at all, untouched rows as they were.
/// No answer names the rewritten fragment.
#[rstest]
#[case::btree(btree())]
#[tokio::test]
async fn answers_with_current_addresses(#[case] case: Case) {
    let store_dir = TempObjDir::default();
    let (remapper, script) = script().into_remapper();
    let index = load(&case, &store_dir, Some(remapper)).await;
    let queries = (0..8).chain(100..104);
    for value in queries.clone() {
        let (rows, is_exact) = search(index.as_ref(), value).await;
        assert!(
            rows.get(&0).is_none(),
            "value {value} answered with a stale address: {rows:?}"
        );
        let expected: RowAddrTreeMap = current_address(value).into_iter().collect();
        if is_exact {
            assert_eq!(rows, expected, "value {value}");
        } else {
            assert!(
                expected
                    .row_addrs()
                    .unwrap()
                    .all(|addr| rows.contains(addr.into())),
                "value {value}: {rows:?} misses {expected:?}"
            );
        }
    }
    assert!(script.batches() > 0, "the index never translated");
    assert!(
        script.batches() < script.addresses(),
        "{} batches for {} addresses: the index translates row by row",
        script.batches(),
        script.addresses()
    );
}

/// Without a remapper the index answers with the addresses it was built
/// over.
#[rstest]
#[case::btree(btree())]
#[tokio::test]
async fn answers_with_built_addresses_without_a_remapper(#[case] case: Case) {
    let store_dir = TempObjDir::default();
    let index = load(&case, &store_dir, None).await;
    for (value, addr) in rows() {
        let (rows, is_exact) = search(index.as_ref(), value).await;
        if is_exact {
            assert_eq!(rows, RowAddrTreeMap::from_iter([addr]), "value {value}");
        } else {
            assert!(rows.contains(addr), "value {value}: {rows:?}");
        }
    }
}
