// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::ScalarValue;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::ROW_ADDR;
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempObjDir;
use lance_io::object_store::ObjectStore;
use roaring::RoaringBitmap;

use super::{StoredZones, matches_target_zones};
use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::registry::VALUE_COLUMN_NAME;
use crate::scalar::zoned::ZoneBound;
use crate::scalar::zonemap::{
    ReusedZones, ScannedZones, ZoneMapIndex, ZoneMapIndexBuilder, ZoneMapIndexBuilderParams,
    ZoneMapIndexVersion, ZoneMapSegmentBase, ZoneMapStatistics, write_zonemap_segment,
};

const ROWS_PER_ZONE: u64 = 4;

fn record(fragment_id: u64, start: u64, length: usize) -> ZoneMapStatistics {
    ZoneMapStatistics {
        min: ScalarValue::Float64(Some(0.0)),
        max: ScalarValue::Float64(Some(1.0)),
        null_count: 0,
        nan_count: 0,
        negative_nan_count: Some(0),
        bound: ZoneBound {
            fragment_id,
            start,
            length,
        },
    }
}

fn records(fragment_id: u64, bounds: &[(u64, usize)]) -> Vec<ZoneMapStatistics> {
    bounds
        .iter()
        .map(|(start, length)| record(fragment_id, *start, *length))
        .collect()
}

#[test]
fn shape_check_accepts_exactly_the_target_zones() {
    // Ten rows in zones of four: [0, 4), [4, 8), [8, 10).
    let exact = records(3, &[(0, 4), (4, 4), (8, 2)]);
    assert!(matches_target_zones(&exact, 3, ROWS_PER_ZONE, 10));
    // Order in the file does not matter.
    let shuffled = records(3, &[(8, 2), (0, 4), (4, 4)]);
    assert!(matches_target_zones(&shuffled, 3, ROWS_PER_ZONE, 10));
    // A fragment smaller than one zone.
    assert!(matches_target_zones(
        &records(3, &[(0, 3)]),
        3,
        ROWS_PER_ZONE,
        3
    ));
}

/// (name, record bounds, physical rows)
type ShapeCase = (&'static str, &'static [(u64, usize)], u64);

#[test]
fn shape_check_rejects_everything_else() {
    let cases: &[ShapeCase] = &[
        ("gap", &[(0, 4), (8, 2)], 10),
        ("duplicate", &[(0, 4), (4, 4), (4, 4), (8, 2)], 10),
        ("overlap", &[(0, 4), (3, 5), (8, 2)], 10),
        (
            "two shorter records inside one target interval",
            &[(0, 4), (4, 2), (6, 2), (8, 2)],
            10,
        ),
        (
            "same count, records cut elsewhere",
            &[(0, 3), (3, 5), (8, 2)],
            10,
        ),
        ("tiles the rows with one record", &[(0, 10)], 10),
        ("fewer rows than the fragment", &[(0, 4), (4, 4)], 10),
        ("more rows than the fragment", &[(0, 4), (4, 4), (8, 4)], 10),
        ("empty fragment", &[], 0),
    ];
    for (name, bounds, physical_rows) in cases {
        assert!(
            !matches_target_zones(&records(3, bounds), 3, ROWS_PER_ZONE, *physical_rows),
            "{name}"
        );
    }
    // A record of another fragment never counts.
    let mut foreign = records(3, &[(0, 4), (4, 4)]);
    foreign.push(record(4, 8, 2));
    assert!(!matches_target_zones(&foreign, 3, ROWS_PER_ZONE, 10));
    assert!(!matches_target_zones(&records(3, &[(0, 4)]), 3, 0, 4));
}

fn local_store(dir: &TempObjDir) -> Arc<LanceIndexStore> {
    Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        (**dir).clone(),
        Arc::new(LanceCache::no_cache()),
    ))
}

/// Values of fragment `f`, row `i`: null every third row, NaN every fifth.
fn value(fragment: u64, row: u64) -> Option<f64> {
    match row % 15 {
        0 | 3 | 6 | 9 | 12 => None,
        5 | 10 => Some(f64::NAN),
        _ => Some((fragment * 100 + row) as f64),
    }
}

/// Write a version 1 segment over `fragments` (id, rows), zones of four.
async fn write_segment(store: &LanceIndexStore, fragments: &[(u64, u64)], keep_nulls: bool) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Float64, true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let batches: Vec<_> = fragments
        .iter()
        .map(|(fragment, rows)| {
            Ok(RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Float64Array::from_iter(
                        (0..*rows).map(|row| value(*fragment, row)),
                    )),
                    Arc::new(UInt64Array::from_iter_values(
                        (0..*rows).map(|row| (fragment << 32) | row),
                    )),
                ],
            )
            .unwrap())
        })
        .collect();
    let mut builder = ZoneMapIndexBuilder::try_new(
        ZoneMapIndexBuilderParams::new(ROWS_PER_ZONE).with_null_bitmap(keep_nulls),
        DataType::Float64,
    )
    .unwrap();
    builder
        .train(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches),
        )))
        .await
        .unwrap();
    builder.write_index(store).await.unwrap();
}

#[tokio::test]
async fn raw_loader_returns_the_requested_fragments_as_the_index_load_does() {
    let dir = TempObjDir::default();
    let store = local_store(&dir);
    write_segment(&store, &[(0, 10), (1, 7), (2, 13)], true).await;

    let wanted = RoaringBitmap::from_iter([0u32, 2]);
    let stored = StoredZones::load(store.as_ref(), ZoneMapIndexVersion::V1, &wanted)
        .await
        .unwrap();
    let index = ZoneMapIndex::load(
        store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    assert_eq!(stored.version(), ZoneMapIndexVersion::V1);
    assert_eq!(stored.value_type(), &DataType::Float64);
    assert_eq!(stored.rows_per_zone(), ROWS_PER_ZONE);
    assert!(stored.store_null_bitmap());
    assert!(stored.has_null_bitmap());
    assert!(stored.decoded_bytes() > 0);
    assert_eq!(
        stored.incompatibility(&DataType::Float64, ROWS_PER_ZONE, true),
        None
    );

    // Exactly the zones the index exposes for fragments 0 and 2, nothing of 1.
    let mut keys: Vec<u64> = stored.zones.keys().copied().collect();
    keys.sort_unstable();
    assert_eq!(keys, vec![0, 2]);
    for fragment in [0u64, 2] {
        let expected: Vec<&ZoneMapStatistics> = index
            .zones
            .iter()
            .filter(|zone| zone.bound.fragment_id == fragment)
            .collect();
        let got: Vec<&ZoneMapStatistics> = stored.zones[&fragment].iter().collect();
        assert_eq!(got, expected, "fragment {fragment}");
    }
    let index_nulls = index.null_rows.as_ref().unwrap();
    let stored_nulls = stored.null_rows.as_ref().unwrap();
    for fragment in [0u32, 2] {
        assert_eq!(
            stored_nulls.get_fragment_bitmap(fragment),
            index_nulls.get_fragment_bitmap(fragment),
            "fragment {fragment}"
        );
        assert!(stored_nulls.get_fragment_bitmap(fragment).is_some());
    }
    assert!(stored_nulls.get(&1).is_none());
}

#[tokio::test]
async fn stored_zones_are_written_back_unchanged() {
    let dir = TempObjDir::default();
    let store = local_store(&dir);
    write_segment(&store, &[(0, 10), (1, 7)], true).await;
    let wanted = RoaringBitmap::from_iter([0u32, 1]);
    let stored = StoredZones::load(store.as_ref(), ZoneMapIndexVersion::V1, &wanted)
        .await
        .unwrap();

    let mut reused = ReusedZones::new(true);
    assert!(reused.reuse_stored(&stored, 0, 10).unwrap());
    // Fragment 1 now holds more rows than the segment summarized.
    assert!(!reused.reuse_stored(&stored, 1, 9).unwrap());
    // A fragment the segment does not store.
    assert!(!reused.reuse_stored(&stored, 5, 4).unwrap());
    assert_eq!(reused.stored_fragments, vec![0]);
    assert_eq!(reused.stored_records, 3);

    let out_dir = TempObjDir::default();
    let out = local_store(&out_dir);
    write_zonemap_segment(
        ZoneMapSegmentBase::New(ZoneMapIndexBuilderParams::new(ROWS_PER_ZONE)),
        &DataType::Float64,
        reused,
        ScannedZones::default(),
        &[0],
        out.as_ref(),
    )
    .await
    .unwrap();
    let source = ZoneMapIndex::load(
        store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    let written = ZoneMapIndex::load(
        out.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    let source_zones: Vec<&ZoneMapStatistics> = source
        .zones
        .iter()
        .filter(|zone| zone.bound.fragment_id == 0)
        .collect();
    let written_zones: Vec<&ZoneMapStatistics> = written.zones.iter().collect();
    assert_eq!(written_zones, source_zones);
    assert_eq!(
        written.null_rows.as_ref().unwrap().get_fragment_bitmap(0),
        source.null_rows.as_ref().unwrap().get_fragment_bitmap(0)
    );
    assert!(written.null_rows.as_ref().unwrap().get(&1).is_none());
}

#[tokio::test]
async fn incompatible_sources_are_named() {
    let dir = TempObjDir::default();
    let store = local_store(&dir);
    write_segment(&store, &[(0, 8)], false).await;
    let wanted = RoaringBitmap::from_iter([0u32]);
    let stored = StoredZones::load(store.as_ref(), ZoneMapIndexVersion::V1, &wanted)
        .await
        .unwrap();
    assert!(!stored.store_null_bitmap());
    assert!(!stored.has_null_bitmap());
    // Same configuration: compatible.
    assert_eq!(
        stored.incompatibility(&DataType::Float64, ROWS_PER_ZONE, false),
        None
    );
    for (data_type, rows_per_zone, keep) in [
        (DataType::Float32, ROWS_PER_ZONE, false),
        (DataType::Float64, ROWS_PER_ZONE * 2, false),
        (DataType::Float64, ROWS_PER_ZONE, true),
    ] {
        assert!(
            stored
                .incompatibility(&data_type, rows_per_zone, keep)
                .is_some(),
            "{data_type:?} {rows_per_zone} {keep}"
        );
    }
    // Without positions an accumulator that keeps them takes nothing.
    let mut reused = ReusedZones::new(true);
    assert!(!reused.reuse_stored(&stored, 0, 8).unwrap());
    let mut reused = ReusedZones::new(false);
    assert!(reused.reuse_stored(&stored, 0, 8).unwrap());

    // The same file read as version 0 is never a current-version source.
    let as_v0 = StoredZones::load(store.as_ref(), ZoneMapIndexVersion::V0, &wanted)
        .await
        .unwrap();
    assert!(
        as_v0
            .incompatibility(&DataType::Float64, ROWS_PER_ZONE, false)
            .is_some()
    );
}
