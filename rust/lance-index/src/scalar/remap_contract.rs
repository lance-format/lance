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

use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, Int32Array, ListArray, RecordBatch, StringArray, UInt64Array};
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

use super::bitmap::BitmapIndexPlugin;
use super::bloomfilter::BloomFilterIndexPlugin;
use super::btree::BTreeIndexPlugin;
use super::inverted::InvertedIndexPlugin;
use super::label_list::LabelListIndexPlugin;
use super::lance_format::LanceIndexStore;
use super::ngram::NGramIndexPlugin;
use super::registry::{ScalarIndexPlugin, TrainingOrdering, VALUE_COLUMN_NAME};
use super::zonemap::ZoneMapIndexPlugin;
use super::{
    AnyQuery, BloomFilterQuery, LabelListQuery, Remapper, SargableQuery, ScalarIndex, TextQuery,
    TokenQuery,
};
use crate::metrics::NoOpMetricsCollector;
use crate::progress::noop_progress;

fn address(fragment: u32, offset: u32) -> u64 {
    RowAddress::new_from_parts(fragment, offset).into()
}

/// Rows the index is built over: value `v` at `address(0, v)` and
/// `address(0, v + 8)` for `v` in `0..8`, the fragment the script rewrites,
/// and value `100 + v` at `address(2, v)` for `v` in `0..4`, a fragment it
/// leaves alone. Each rewritten value has two rows, so an index that
/// translates a value's rows together is told apart from one that
/// translates row by row.
fn rows() -> Vec<(i32, u64)> {
    let rewritten = (0..16).map(|offset| (offset as i32 % 8, address(0, offset)));
    let untouched = (0..4).map(|value| (100 + value, address(2, value as u32)));
    rewritten.chain(untouched).collect()
}

/// Offsets of fragment 0 whose rows the script deletes: one row of value
/// 3, one of value 5, and both rows of value 6.
const DELETED: [u32; 4] = [3, 13, 6, 14];

/// Fragment 0 was rewritten into fragment 1 in reverse order, and the rows
/// at [`DELETED`] were deleted.
fn script() -> ScriptedRemapper {
    (0..16u32).fold(ScriptedRemapper::new(), |script, offset| {
        if DELETED.contains(&offset) {
            script.deleted(address(0, offset))
        } else {
            script.moved(address(0, offset), address(1, 15 - offset))
        }
    })
}

/// The rows holding `value` that the index was built over.
fn built_addresses(value: i32) -> RowAddrTreeMap {
    rows()
        .into_iter()
        .filter(|&(row_value, _)| row_value == value)
        .map(|(_, addr)| addr)
        .collect()
}

/// Where the rows holding `value` are after the script.
fn current_addresses(value: i32) -> RowAddrTreeMap {
    built_addresses(value)
        .row_addrs()
        .unwrap()
        .map(u64::from)
        .filter_map(|addr| {
            let offset = RowAddress::from(addr).row_offset();
            match RowAddress::from(addr).fragment_id() {
                0 if DELETED.contains(&offset) => None,
                0 => Some(address(1, 15 - offset)),
                _ => Some(addr),
            }
        })
        .collect()
}

/// One index kind under test.
struct Case {
    plugin: Arc<dyn ScalarIndexPlugin>,
    params: &'static str,
    /// The fragment IDs handed to training, for kinds that record them.
    fragment_ids: Option<Vec<u32>>,
    /// The type of the indexed column; see [`value_array`].
    data_type: DataType,
    /// The query this kind answers for the rows holding a value.
    query: fn(i32) -> Box<dyn AnyQuery>,
}

fn equals(value: i32) -> Box<dyn AnyQuery> {
    Box::new(SargableQuery::Equals(ScalarValue::Int32(Some(value))))
}

/// The text a string column holds for `value`: no value's text contains
/// another's.
fn text(value: i32) -> String {
    format!("<{value:03}>")
}

/// The indexed column for `values`, of type `data_type`.
fn value_array(data_type: &DataType, values: Vec<i32>) -> ArrayRef {
    match data_type {
        DataType::Int32 => Arc::new(Int32Array::from(values)),
        DataType::Utf8 => Arc::new(StringArray::from_iter_values(values.into_iter().map(text))),
        // Each row's list holds its one value.
        DataType::List(_) => Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
            values.into_iter().map(|value| Some([Some(value)])),
        )),
        other => panic!("no rows of type {other}"),
    }
}

fn btree() -> Case {
    Case {
        plugin: Arc::new(BTreeIndexPlugin),
        params: "{}",
        fragment_ids: Some(vec![0, 2]),
        data_type: DataType::Int32,
        query: equals,
    }
}

fn bitmap() -> Case {
    Case {
        plugin: Arc::new(BitmapIndexPlugin),
        params: "{}",
        // A bitmap build given several fragments expects a shard ID.
        fragment_ids: None,
        data_type: DataType::Int32,
        query: equals,
    }
}

fn zonemap() -> Case {
    Case {
        plugin: Arc::new(ZoneMapIndexPlugin),
        params: r#"{"rows_per_zone": 4}"#,
        fragment_ids: Some(vec![0, 2]),
        data_type: DataType::Int32,
        query: equals,
    }
}

fn bloomfilter() -> Case {
    Case {
        plugin: Arc::new(BloomFilterIndexPlugin),
        params: r#"{"number_of_items": 4, "probability": 0.01}"#,
        fragment_ids: Some(vec![0, 2]),
        data_type: DataType::Int32,
        query: |value| Box::new(BloomFilterQuery::Equals(ScalarValue::Int32(Some(value)))),
    }
}

fn ngram() -> Case {
    Case {
        plugin: Arc::new(NGramIndexPlugin),
        params: "{}",
        fragment_ids: Some(vec![0, 2]),
        data_type: DataType::Utf8,
        query: |value| Box::new(TextQuery::StringContains(text(value))),
    }
}

fn inverted() -> Case {
    Case {
        plugin: Arc::new(InvertedIndexPlugin),
        params: "{}",
        // Training over named fragments is a distributed build, which
        // leaves the metadata file to the merge.
        fragment_ids: None,
        data_type: DataType::Utf8,
        query: |value| Box::new(TokenQuery::TokensContains(text(value))),
    }
}

fn label_list() -> Case {
    Case {
        plugin: Arc::new(LabelListIndexPlugin),
        params: "{}",
        // Like a bitmap build, a label list build given several fragments
        // expects a shard ID.
        fragment_ids: None,
        data_type: DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        query: |value| {
            Box::new(LabelListQuery::HasAnyLabel(vec![ScalarValue::Int32(Some(
                value,
            ))]))
        },
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
            &Field::new(VALUE_COLUMN_NAME, case.data_type.clone(), true),
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
    let mut fields = vec![Field::new(VALUE_COLUMN_NAME, case.data_type.clone(), true)];
    let mut columns: Vec<ArrayRef> = vec![value_array(&case.data_type, values)];
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
            case.fragment_ids.clone(),
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

async fn search(case: &Case, index: &dyn ScalarIndex, value: i32) -> (RowAddrTreeMap, bool) {
    let result = index
        .search((case.query)(value).as_ref(), &NoOpMetricsCollector)
        .await
        .unwrap();
    (result.row_addrs().true_rows(), result.is_exact())
}

/// Each value's rows come back at their current addresses: moved rows at
/// their destination, deleted rows not at all, untouched rows as they were.
/// No answer names the rewritten fragment.
#[rstest]
#[case::btree(btree())]
#[case::bitmap(bitmap())]
#[case::zonemap(zonemap())]
#[case::bloomfilter(bloomfilter())]
#[case::ngram(ngram())]
#[case::inverted(inverted())]
#[case::label_list(label_list())]
#[tokio::test]
async fn answers_with_current_addresses(#[case] case: Case) {
    let store_dir = TempObjDir::default();
    let (remapper, script) = script().into_remapper();
    let index = load(&case, &store_dir, Some(remapper)).await;
    let queries = (0..8).chain(100..104);
    for value in queries.clone() {
        let (rows, is_exact) = search(&case, index.as_ref(), value).await;
        assert!(
            rows.get(&0).is_none(),
            "value {value} answered with a stale address: {rows:?}"
        );
        let expected = current_addresses(value);
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
#[case::bitmap(bitmap())]
#[case::zonemap(zonemap())]
#[case::bloomfilter(bloomfilter())]
#[case::ngram(ngram())]
#[case::inverted(inverted())]
#[case::label_list(label_list())]
#[tokio::test]
async fn answers_with_built_addresses_without_a_remapper(#[case] case: Case) {
    let store_dir = TempObjDir::default();
    let index = load(&case, &store_dir, None).await;
    for value in (0..8).chain(100..104) {
        let (rows, is_exact) = search(&case, index.as_ref(), value).await;
        let expected = built_addresses(value);
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
}
