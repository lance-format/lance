// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Unit tests for the ZoneMap scalar index.
//!
//! These live in their own module so `zonemap.rs` stays under the module
//! size limit enforced by `ci/check_module_size.py`.

use crate::scalar::registry::VALUE_COLUMN_NAME;
use crate::scalar::{IndexStore, zonemap::ROWS_PER_ZONE_DEFAULT};
use std::sync::Arc;

use crate::scalar::registry::ScalarIndexPlugin;
use crate::scalar::zoned::ZoneBound;
use crate::scalar::zonemap::{
    CarriedSeed, MergeableSeed, ZoneMapIndexPlugin, ZoneMapIndexVersion, ZoneMapStatistics,
    make_zone_map_index_details,
};
use arrow::datatypes::{ArrowPrimitiveType, Decimal128Type, Float32Type, Float64Type, Int64Type};
use arrow_array::{
    Array, Float64Array, PrimitiveArray, RecordBatch, UInt32Array, UInt64Array, record_batch,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::ScalarValue;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::{StreamExt, TryStreamExt, stream};
use lance_core::utils::tempfile::TempObjDir;
use lance_core::{
    ROW_ADDR,
    cache::{LanceCache, WeakLanceCache},
};
use lance_datafusion_exec::datagen::DatafusionDatagenExt;
use lance_datagen::ArrayGeneratorExt;
use lance_datagen::{BatchCount, RowCount, array};
use lance_io::object_store::ObjectStore;
use lance_select::{RowAddrTreeMap, RowSetOps};
use std::ops::Bound;

use crate::pbold;
use crate::scalar::CreatedIndex;
use crate::scalar::seed::IndexSeedWriter;
use crate::scalar::zoned::ZoneProcessor;
use crate::scalar::{
    RowIdRemapper, SargableQuery, ScalarIndex, SearchResult,
    lance_format::LanceIndexStore,
    zonemap::{
        HarvestedZoneSeed, ReusedZones, SEED_NULL_BITMAP_META_KEY, SEED_VERSION_META_KEY,
        SeedOutcome, ZONEMAP_FILENAME, ZONEMAP_SIZE_META_KEY, ZoneMapIndex,
        ZoneMapIndexBuilderParams, ZoneMapProcessor, ZoneMapSeedWriter, ZoneMapSegmentBase,
        build_zonemap_segment, merge_seeds_for_binary_copy, merge_zonemap_indices,
        parse_seed_for_binary_copy, place_records, remap_zone,
    },
};
use arrow_array::Int32Array;
use lance_core::Error;

use crate::Index;
use crate::metrics::NoOpMetricsCollector;
use roaring::RoaringBitmap;
use std::collections::HashMap;

// Adds a _rowaddr column emulating each batch as a new fragment
fn add_row_addr(stream: SendableRecordBatchStream) -> SendableRecordBatchStream {
    let schema = stream.schema();
    let schema_with_row_addr = Arc::new(Schema::new(vec![
        schema.field(0).clone(),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let schema = schema_with_row_addr.clone();
    let stream = stream.enumerate().map(move |(frag_id, batch)| {
        let batch = batch.unwrap();
        let row_addr = Arc::new(UInt64Array::from_iter_values(
            (0..batch.num_rows() as u64).map(|off| off + ((frag_id as u64) << 32)),
        ));
        Ok(RecordBatch::try_new(
            schema_with_row_addr.clone(),
            vec![batch.column(0).clone(), row_addr],
        )?)
    });
    Box::pin(RecordBatchStreamAdapter::new(schema, stream))
}

/// Build a single-column ZoneMap of primitive type `T` from `fragments`
/// (one batch -> one fragment), with small zones, then load it back.
async fn train_and_load<T: ArrowPrimitiveType>(
    fragments: Vec<Vec<Option<T::Native>>>,
) -> Arc<ZoneMapIndex>
where
    PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
{
    // The directory goes away with this call; the loaded zones are in
    // memory, so only tests that reopen the file need `train_and_load_in`.
    let tmpdir = TempObjDir::default();
    train_and_load_in::<T>(local_store(&tmpdir), fragments).await
}

/// Trains a version 1 index over `fragments` (two rows per zone) into
/// `test_store` and loads it back.
async fn train_and_load_in<T: ArrowPrimitiveType>(
    test_store: Arc<LanceIndexStore>,
    fragments: Vec<Vec<Option<T::Native>>>,
) -> Arc<ZoneMapIndex>
where
    PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
{
    let schema = Arc::new(Schema::new(vec![Field::new(
        VALUE_COLUMN_NAME,
        T::DATA_TYPE,
        true,
    )]));
    let batches: Vec<RecordBatch> = fragments
        .into_iter()
        .map(|vals| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(PrimitiveArray::<T>::from(vals))],
            )
            .unwrap()
        })
        .collect();
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter(batches.into_iter().map(Ok)),
    ));
    let stream = add_row_addr(stream);

    ZoneMapIndexPlugin::train_zonemap_index(
        stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(2)),
    )
    .await
    .unwrap();

    ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .expect("Failed to load ZoneMapIndex")
}

#[derive(Debug)]
struct TestRemapper {
    mappings: HashMap<u64, u64>,
}

impl TestRemapper {
    fn new(mappings: impl IntoIterator<Item = (u64, u64)>) -> Self {
        Self {
            mappings: mappings.into_iter().collect(),
        }
    }
}

impl RowIdRemapper for TestRemapper {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        self.mappings.get(&row_id).copied()
    }

    fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        self.mappings
            .iter()
            .filter(|(old, _)| row_addrs.contains(**old))
            .map(|(_, new)| *new)
            .collect()
    }

    fn remap_row_ids_roaring_tree_map(
        &self,
        _row_ids: &roaring::RoaringTreemap,
    ) -> roaring::RoaringTreemap {
        unreachable!()
    }

    fn remap_row_ids_record_batch(
        &self,
        _batch: RecordBatch,
        _row_id_idx: usize,
    ) -> lance_core::Result<RecordBatch> {
        unreachable!()
    }
}

#[test]
fn test_remap_zone_splits_discontiguous_runs() {
    let old_start = (2_u64 << 32) + 10;
    let new_start = 3_u64 << 32;
    let zone = ZoneMapStatistics {
        min: ScalarValue::Int64(Some(10)),
        max: ScalarValue::Int64(Some(40)),
        null_count: 1,
        nan_count: 0,
        negative_nan_count: Some(0),
        bound: ZoneBound {
            fragment_id: 2,
            start: 10,
            length: 4,
        },
    };
    let mut first_run = zone.clone();
    first_run.bound = ZoneBound {
        fragment_id: 3,
        start: 1,
        length: 2,
    };
    let mut second_run = zone.clone();
    second_run.bound = ZoneBound {
        fragment_id: 3,
        start: 7,
        length: 2,
    };
    second_run.null_count = 0;

    let remapper = TestRemapper::new([
        (old_start, new_start + 1),
        (old_start + 1, new_start + 2),
        (old_start + 2, new_start + 7),
        (old_start + 3, new_start + 8),
    ]);
    let mut remapped_null_rows = RowAddrTreeMap::new();
    remapped_null_rows.insert(new_start + 1);

    assert_eq!(
        remap_zone(&zone, &remapper, Some(&remapped_null_rows), false, None).unwrap(),
        vec![first_run, second_run]
    );
}

#[tokio::test]
async fn test_remap_nested_mixed_null_zone_keeps_non_null_candidates() {
    let index = train_and_load_fsl(
        vec![
            vec![Some(0.0), Some(0.0)],
            vec![Some(0.0), Some(0.0)],
            vec![Some(3.0), Some(4.0)],
            vec![Some(5.0), Some(6.0)],
        ],
        2,
    )
    .await;
    let mut mixed_null_zone = index.zones[0].clone();
    mixed_null_zone.null_count = 2;

    let new_start = 3_u64 << 32;
    let remapper = TestRemapper::new([(2, new_start), (3, new_start + 1)]);
    let remapped_null_rows = RowAddrTreeMap::new();
    let zones = remap_zone(
        &mixed_null_zone,
        &remapper,
        Some(&remapped_null_rows),
        true,
        None,
    )
    .unwrap();

    assert_eq!(zones.len(), 1);
    assert_eq!(zones[0].bound.length, 2);
    assert_eq!(zones[0].null_count, 0);
    assert!(
        index
            .evaluate_zone_against_query(
                &zones[0],
                &SargableQuery::Equals(fsl_scalar(vec![Some(3.0), Some(4.0)])),
            )
            .unwrap(),
        "the two surviving rows are non-null candidates"
    );
}

#[test]
fn test_remap_nested_mixed_null_singleton_without_bitmap_is_rejected() {
    let zone = ZoneMapStatistics {
        min: ScalarValue::Null,
        max: ScalarValue::Null,
        null_count: 2,
        nan_count: 0,
        negative_nan_count: Some(0),
        bound: ZoneBound {
            fragment_id: 0,
            start: 0,
            length: 4,
        },
    };
    let remapper = TestRemapper::new([(2, 0)]);

    let error = remap_zone(&zone, &remapper, None, true, None).unwrap_err();
    assert!(matches!(error, lance_core::Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("cannot safely remap a mixed-null ZoneMap zone")
    );
}

#[tokio::test]
async fn test_merge_legacy_decimal_mixed_null_singleton_is_rejected() {
    let mut source = train_and_load::<Decimal128Type>(vec![vec![None, Some(200)]]).await;
    let source_mut = Arc::get_mut(&mut source).unwrap();
    let zone = &mut source_mut.zones[0];
    zone.min = ScalarValue::Decimal128(None, 38, 10);
    zone.max = ScalarValue::Decimal128(None, 38, 10);
    source_mut.null_rows = None;
    source_mut.fri = Some(Arc::new(TestRemapper::new([(1, 3_u64 << 32)])));

    let candidate = ScalarValue::Decimal128(Some(200), 38, 10);
    let queries = [
        SargableQuery::Equals(candidate.clone()),
        SargableQuery::Range(
            Bound::Included(candidate.clone()),
            Bound::Included(candidate.clone()),
        ),
        SargableQuery::IsIn(vec![candidate]),
    ];
    for query in queries {
        assert!(
            source
                .evaluate_zone_against_query(&source.zones[0], &query)
                .unwrap(),
            "legacy Decimal source must retain the non-null candidate for {query:?}"
        );
    }

    let dest_tmpdir = TempObjDir::default();
    let dest_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        dest_tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let error = merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([3]),
    )
    .await
    .err()
    .expect("ambiguous legacy Decimal singleton must reject consolidation");

    assert!(matches!(error, lance_core::Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("cannot safely remap a mixed-null ZoneMap zone")
    );
}

/// Remapping copies the source zone's counts into a shorter run, so a
/// legacy NaN-marker zone [5.0, +NaN] whose NaN row is removed must still
/// answer bounded ranges around 5.0 from its bounds, directly and after
/// a version 0 merge that persists the remapped zone.
#[tokio::test]
async fn test_remapped_legacy_nan_marker_zone_keeps_bounded_range_candidate() {
    let mut source = train_and_load::<Float64Type>(vec![vec![Some(5.0), Some(f64::NAN)]]).await;
    let source_mut = Arc::get_mut(&mut source).unwrap();
    source_mut.version = ZoneMapIndexVersion::V0;
    let zone = &mut source_mut.zones[0];
    zone.max = ScalarValue::Float64(Some(f64::NAN));
    zone.negative_nan_count = None;
    assert_eq!(zone.nan_count, 1);
    source_mut.null_rows = None;
    let new_row = 3_u64 << 32;
    source_mut.fri = Some(Arc::new(TestRemapper::new([(0, new_row)])));

    let remapped = remap_zone(
        &source.zones[0],
        source.fri.as_deref().unwrap(),
        None,
        false,
        None,
    )
    .unwrap();
    assert_eq!(remapped.len(), 1);
    assert_eq!(remapped[0].bound.length, 1);
    assert_eq!(remapped[0].nan_count, 1, "inherited, not exact");

    let five = ScalarValue::Float64(Some(5.0));
    let queries = [
        SargableQuery::Range(Bound::Included(five.clone()), Bound::Included(five.clone())),
        SargableQuery::Range(
            Bound::Included(ScalarValue::Float64(Some(4.0))),
            Bound::Included(ScalarValue::Float64(Some(6.0))),
        ),
        SargableQuery::Range(
            Bound::Excluded(ScalarValue::Float64(Some(4.0))),
            Bound::Excluded(ScalarValue::Float64(Some(6.0))),
        ),
        SargableQuery::Equals(five.clone()),
    ];
    for query in &queries {
        assert!(
            source
                .evaluate_zone_against_query(&remapped[0], query)
                .unwrap(),
            "the surviving 5.0 row must stay a candidate for {query:?}"
        );
    }

    let dest_tmpdir = TempObjDir::default();
    let dest_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        dest_tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let created = merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([3]),
    )
    .await
    .unwrap();
    assert_eq!(created.index_version, 0);
    let merged = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V0,
    )
    .await
    .unwrap();
    for query in &queries {
        let result = merged.search(query, &NoOpMetricsCollector).await.unwrap();
        assert!(
            result.row_addrs().selected(new_row),
            "merged index must keep the 5.0 row for {query:?}"
        );
    }
}

/// The legacy-null variant: [5, null] without a null bitmap remapped to
/// the single 5 row keeps `null_count = 1`; the bounds, not that count,
/// decide bounded ranges. Such an inexact count is persisted only by a
/// version 0 merge (a version 1 merge refuses it).
#[tokio::test]
async fn test_remapped_legacy_null_zone_keeps_bounded_range_candidate() {
    let mut source = train_and_load::<Int64Type>(vec![vec![Some(5), None]]).await;
    let source_mut = Arc::get_mut(&mut source).unwrap();
    source_mut.version = ZoneMapIndexVersion::V0;
    source_mut.null_rows = None;
    let new_row = 3_u64 << 32;
    source_mut.fri = Some(Arc::new(TestRemapper::new([(0, new_row)])));

    let remapped = remap_zone(
        &source.zones[0],
        source.fri.as_deref().unwrap(),
        None,
        false,
        None,
    )
    .unwrap();
    assert_eq!(remapped.len(), 1);
    assert_eq!(remapped[0].bound.length, 1);
    assert_eq!(remapped[0].null_count, 1, "inherited, not exact");

    let five = ScalarValue::Int64(Some(5));
    let queries = [
        SargableQuery::Range(Bound::Included(five.clone()), Bound::Included(five.clone())),
        SargableQuery::Range(
            Bound::Included(ScalarValue::Int64(Some(4))),
            Bound::Excluded(ScalarValue::Int64(Some(6))),
        ),
        SargableQuery::Equals(five.clone()),
    ];
    for query in &queries {
        assert!(
            source
                .evaluate_zone_against_query(&remapped[0], query)
                .unwrap(),
            "the surviving 5 row must stay a candidate for {query:?}"
        );
    }

    let dest_tmpdir = TempObjDir::default();
    let dest_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        dest_tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([3]),
    )
    .await
    .unwrap();
    let merged = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V0,
    )
    .await
    .unwrap();
    for query in &queries {
        let result = merged.search(query, &NoOpMetricsCollector).await.unwrap();
        assert!(
            result.row_addrs().selected(new_row),
            "merged index must keep the 5 row for {query:?}"
        );
    }
}

#[tokio::test]
async fn test_value_range_spans_fragments() {
    // Two fragments, multiple zones each; global min/max straddle both.
    let index = train_and_load::<Int64Type>(vec![
        vec![Some(10), Some(50), Some(30)],
        vec![Some(5), Some(99), Some(42)],
    ])
    .await;
    assert_eq!(
        index.value_range(),
        Some((ScalarValue::Int64(Some(5)), ScalarValue::Int64(Some(99))))
    );
}

#[tokio::test]
async fn test_value_range_all_null_is_none() {
    let index = train_and_load::<Int64Type>(vec![vec![None, None, None]]).await;
    assert_eq!(index.value_range(), None);

    let result = index
        .search(
            &SargableQuery::Equals(ScalarValue::Int64(Some(1))),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));
}

#[tokio::test]
async fn test_value_range_nan_max_is_none() {
    // Zones of size 2: [1.0, 2.0] then [100.0, NaN]. The NaN-bearing zone hides
    // its finite max (100.0), so the only sound answer is None.
    let index = train_and_load::<Float32Type>(vec![vec![
        Some(1.0),
        Some(2.0),
        Some(100.0),
        Some(f32::NAN),
    ]])
    .await;
    assert_eq!(index.value_range(), None);
}

#[tokio::test]
async fn test_value_range_over_folds_segments() {
    // Two disjoint segments of one logical index; the global range straddles
    // both (min and max from `b`), proving the fold spans segments.
    let a = train_and_load::<Int64Type>(vec![vec![Some(5), Some(9)]]).await;
    let b = train_and_load::<Int64Type>(vec![vec![Some(1), Some(20)]]).await;
    assert_eq!(
        ZoneMapIndex::value_range_over([a.as_ref(), b.as_ref()]),
        Some((ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(20))))
    );
}

#[tokio::test]
async fn test_value_range_over_nan_in_any_segment_is_none() {
    // NaN in one segment hides that segment's finite max; the cross-segment
    // fold must bail to None just as the single-segment path does.
    let a = train_and_load::<Float32Type>(vec![vec![Some(1.0), Some(2.0)]]).await;
    let b = train_and_load::<Float32Type>(vec![vec![Some(100.0), Some(f32::NAN)]]).await;
    assert_eq!(
        ZoneMapIndex::value_range_over([a.as_ref(), b.as_ref()]),
        None
    );
}

#[tokio::test]
async fn test_value_range_over_missing_extrema() {
    // An all-null segment yields no finite zone; folding it with a finite
    // segment returns the finite segment's range (null contributes nothing).
    let a = train_and_load::<Int64Type>(vec![vec![None, None]]).await;
    let b = train_and_load::<Int64Type>(vec![vec![Some(3), Some(7)]]).await;
    assert_eq!(
        ZoneMapIndex::value_range_over([a.as_ref(), b.as_ref()]),
        Some((ScalarValue::Int64(Some(3)), ScalarValue::Int64(Some(7))))
    );

    // Lance v8 persisted typed-null Decimal extrema even when a zone contained
    // values. Such unknown bounds cannot be skipped like an all-null segment.
    let mut legacy = train_and_load::<Decimal128Type>(vec![vec![Some(100), Some(200)]]).await;
    for zone in &mut Arc::get_mut(&mut legacy).unwrap().zones {
        zone.min = ScalarValue::Decimal128(None, 38, 10);
        zone.max = ScalarValue::Decimal128(None, 38, 10);
    }
    let current = train_and_load::<Decimal128Type>(vec![vec![Some(10_000), Some(20_000)]]).await;
    assert_eq!(
        ZoneMapIndex::value_range_over([legacy.as_ref(), current.as_ref()]),
        None
    );
}

/// Indices that do not override `lookup` refuse it rather than returning
/// no matches, so callers can fall back to another plan.
#[tokio::test]
async fn test_lookup_is_not_supported() {
    let index = train_and_load::<arrow_array::types::Int32Type>(vec![vec![Some(1), Some(2)]]).await;
    let keys = Arc::new(arrow_array::Int32Array::from(vec![1]));
    let err = index.lookup(keys, &NoOpMetricsCollector).err().unwrap();
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
}

#[tokio::test]
async fn test_empty_zonemap_index() {
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let data = arrow_array::Int32Array::from(Vec::<i32>::new());
    let row_ids = arrow_array::UInt64Array::from(Vec::<u64>::new());
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Int32, false),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let data =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();

    let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::once(std::future::ready(Ok(data))),
    ));

    ZoneMapIndexPlugin::train_zonemap_index(data_stream, test_store.as_ref(), None)
        .await
        .unwrap();

    log::debug!("Successfully wrote the index file");

    // Read the index file back and check its contents
    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .expect("Failed to load ZoneMapIndex");
    assert_eq!(index.zones.len(), 0);
    assert_eq!(index.data_type, DataType::Int32);
    assert_eq!(index.rows_per_zone, ROWS_PER_ZONE_DEFAULT);

    // Equals query: null (should match nothing, as there are no nulls)
    let query = SargableQuery::Equals(ScalarValue::Int32(None));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));
}

#[tokio::test]
// Test that a zonemap index can be created with null values from few fragments
async fn test_null_zonemap_index() {
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let stream = lance_datagen::gen_batch()
        .col(
            VALUE_COLUMN_NAME,
            array::rand::<Float32Type>().with_nulls(&[true, false, false, false, false]),
        )
        .into_df_stream(RowCount::from(5000), BatchCount::from(10));

    // Add _rowaddr column
    let stream = add_row_addr(stream);

    ZoneMapIndexPlugin::train_zonemap_index(
        stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(5000)),
    )
    .await
    .unwrap();

    log::debug!("Successfully wrote the index file");

    // Read the index file back and check its contents
    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .expect("Failed to load ZoneMapIndex");
    assert_eq!(index.zones.len(), 10);
    for (i, zone) in index.zones.iter().enumerate() {
        assert_eq!(zone.null_count, 1000);
        assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
        assert_eq!(zone.bound.length, 5000);
        assert_eq!(zone.bound.fragment_id, i as u64);
    }

    // Equals query: null (should match all zones since they contain null values)
    let query = SargableQuery::Equals(ScalarValue::Int32(None));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // Create expected RowAddrTreeMap with all zones since they contain null values
    let mut expected = RowAddrTreeMap::new();
    for fragment_id in 0..10 {
        let start = (fragment_id as u64) << 32;
        let end = start + 5000;
        expected.insert_range(start..end);
    }
    assert_eq!(result, SearchResult::at_most(expected));

    // Test update - add new data with Float32 values (matching the original data type)
    let new_data =
        arrow_array::Float32Array::from_iter_values((0..5000).map(|i| i as f32 / 1000.0));
    // Create row addresses for fragment 10 (next fragment after 0-9)
    let new_row_addr = UInt64Array::from_iter_values((0..5000).map(|i| (10u64 << 32) | (i as u64)));
    let new_schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Float32, false), // Match original schema
        Field::new(ROW_ADDR, DataType::UInt64, false), // Use _rowaddr as expected by the builder
    ]));
    let new_data_batch = RecordBatch::try_new(
        new_schema.clone(),
        vec![Arc::new(new_data), Arc::new(new_row_addr)],
    )
    .unwrap();
    let new_data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        new_schema,
        stream::once(std::future::ready(Ok(new_data_batch))),
    ));

    // Directly pass the stream with proper row addresses instead of using MockTrainingSource
    // which would regenerate row addresses starting from 0
    index
        .update(new_data_stream, test_store.as_ref(), None)
        .await
        .unwrap();

    // Verify the updated index has more zones
    let updated_index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .expect("Failed to load updated ZoneMapIndex");

    // Should have original 10 zones + 1 new zone (5000 rows with zone size 5000)
    assert_eq!(updated_index.zones.len(), 11);

    // Verify the new zone was added
    let new_zone = &updated_index.zones[10]; // Last zone should be the new one
    assert_eq!(new_zone.bound.fragment_id, 10u64); // New fragment ID
    assert_eq!(new_zone.bound.length, 5000);
    assert_eq!(new_zone.null_count, 0); // New data has no nulls
    assert_eq!(new_zone.nan_count, 0); // New data has no NaN values

    // Test search on updated index - search for null values should still work
    let query = SargableQuery::Equals(ScalarValue::Float32(None));
    let result = updated_index
        .search(&query, &NoOpMetricsCollector)
        .await
        .unwrap();

    // Should match original 10 zones (with nulls) but not the new zone (no nulls)
    let mut expected = RowAddrTreeMap::new();
    for fragment_id in 0..10 {
        let start = (fragment_id as u64) << 32;
        let end = start + 5000;
        expected.insert_range(start..end);
    }
    assert_eq!(result, SearchResult::at_most(expected));

    // Test search for a value that should be in the new zone
    let query = SargableQuery::Equals(ScalarValue::Float32(Some(2.5))); // Value 2500/1000 = 2.5
    let result = updated_index
        .search(&query, &NoOpMetricsCollector)
        .await
        .unwrap();

    // Should match the new zone (fragment 10)
    let mut expected = RowAddrTreeMap::new();
    let start = 10u64 << 32;
    let end = start + 5000;
    expected.insert_range(start..end);
    assert_eq!(result, SearchResult::at_most(expected));
}

#[tokio::test]
async fn test_zonemap_null_handling_in_queries() {
    // Test that zonemap index correctly returns null_list for queries
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Create test data: [0, 5, null]
    let batch = record_batch!(
        (VALUE_COLUMN_NAME, Int64, [Some(0), Some(5), None]),
        (ROW_ADDR, UInt64, [0, 1, 2])
    )
    .unwrap();
    let schema = batch.schema();
    let stream = stream::once(async move { Ok(batch) });
    let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

    // Train and write the zonemap index
    ZoneMapIndexPlugin::train_zonemap_index(stream, store.as_ref(), None)
        .await
        .unwrap();

    let cache = LanceCache::with_capacity(1024 * 1024);
    let index = ZoneMapIndex::load(store.clone(), None, &cache, false, ZoneMapIndexVersion::V1)
        .await
        .unwrap();

    // Test 1: Search for value 5 - zonemap should return at_most with all rows
    // Since ZoneMap returns AtMost (superset), it's correct to include nulls in the result
    let query = SargableQuery::Equals(ScalarValue::Int64(Some(5)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    match result {
        SearchResult::AtMost(row_ids) => {
            // Zonemap can't determine exact matches, so it returns all rows in the zone
            // This includes nulls because ZoneMap can't prove they don't match
            let all_rows: Vec<u64> = row_ids
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(u64::from)
                .collect();
            assert_eq!(
                all_rows,
                vec![0, 1, 2],
                "Should return all rows (including nulls) since ZoneMap is inexact"
            );

            // For AtMost results, nulls are included in the superset
            // Downstream processing will handle null filtering
        }
        _ => panic!("Expected AtMost search result from zonemap"),
    }

    // Test 2: Range query - should also return all rows as AtMost
    let query = SargableQuery::Range(
        std::ops::Bound::Included(ScalarValue::Int64(Some(0))),
        std::ops::Bound::Included(ScalarValue::Int64(Some(3))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    match result {
        SearchResult::AtMost(row_ids) => {
            // Again, ZoneMap returns superset including nulls
            let all_rows: Vec<u64> = row_ids
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(u64::from)
                .collect();
            assert_eq!(
                all_rows,
                vec![0, 1, 2],
                "Should return all rows in zone as possible matches"
            );
        }
        _ => panic!("Expected AtMost search result from zonemap"),
    }
}

#[tokio::test]
async fn test_nan_zonemap_index() {
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Create deterministic data with NaN values
    // Pattern: [1.0, 2.0, NaN, 3.0, 4.0, 5.0, NaN, 6.0, 7.0, 8.0, ...]
    let mut values = Vec::new();
    for i in 0..500 {
        if i % 5 == 2 {
            values.push(f32::NAN);
        } else {
            // Other values are sequential numbers
            values.push(i as f32);
        }
    }

    let float_data = arrow_array::Float32Array::from(values);
    let row_ids = UInt64Array::from_iter_values((0..float_data.len()).map(|i| i as u64));
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Float32, true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(float_data.clone()), Arc::new(row_ids)],
    )
    .unwrap();
    let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::once(std::future::ready(Ok(data))),
    ));

    ZoneMapIndexPlugin::train_zonemap_index(
        data_stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(100)),
    )
    .await
    .unwrap();

    // Load the index
    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .expect("Failed to load ZoneMapIndex");

    // Should have 5 zones since we have 500 rows and zone size is 100
    assert_eq!(index.zones.len(), 5);

    // Check that each zone has the expected NaN count
    // Each zone has 100 values, and every 5th value (indices 2, 7, 12, ...) is NaN
    // So each zone should have 20 NaN values (100/5 = 20)
    for (i, zone) in index.zones.iter().enumerate() {
        assert_eq!(zone.nan_count, 20, "Zone {} should have 20 NaN values", i);
        assert_eq!(
            zone.bound.length, 100,
            "Zone {} should have zone_length 100",
            i
        );
        assert_eq!(
            zone.bound.fragment_id, 0u64,
            "Zone {} should have fragment_id 0",
            i
        );
    }

    // Version 1 files keep the ordinary bounds and the exact sign split.
    let zone = &index.zones[0];
    assert_eq!(zone.min, ScalarValue::Float32(Some(0.0)));
    assert_eq!(zone.max, ScalarValue::Float32(Some(99.0)));
    assert_eq!(zone.negative_nan_count, Some(0));

    // Test search for NaN values using Equals with NaN
    let query = SargableQuery::Equals(ScalarValue::Float32(Some(f32::NAN)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // Should match all zones since they all contain NaN values
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..500); // All rows since NaN is in every zone
    assert_eq!(result, SearchResult::at_most(expected));

    // The planner uses this raw total-order range to retrieve negative
    // NaNs, which sort below negative infinity. Version 1 records the NaN
    // sign split, so a zone whose NaNs are all positive is pruned here;
    // version 0 has no split to consult and keeps every NaN-bearing zone
    // as a candidate. Every NaN in this data is positive.
    let query = SargableQuery::Range(
        Bound::Unbounded,
        Bound::Excluded(ScalarValue::Float32(Some(f32::NEG_INFINITY))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(
        result,
        SearchResult::at_most(RowAddrTreeMap::new()),
        "version 1 knows every NaN here is positive, so no zone can hold \
         a value below negative infinity"
    );

    // Test search for a specific finite value that exists in the data
    let query = SargableQuery::Equals(ScalarValue::Float32(Some(5.0)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // Should match only the first zone since 5.0 only exists in rows 0-99
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..100);
    assert_eq!(result, SearchResult::at_most(expected));

    // Test search for a value that doesn't exist
    let query = SargableQuery::Equals(ScalarValue::Float32(Some(1000.0)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // The ordinary bounds exclude 1000.0 and NaNs never equal a finite
    // value, so no zone survives.
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

    // Test range query that should include finite values
    let query = SargableQuery::Range(
        Bound::Included(ScalarValue::Float32(Some(0.0))),
        Bound::Included(ScalarValue::Float32(Some(250.0))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // NaNs of either sign fail one side of a two-sided finite range, so
    // they never keep a zone here; the ordinary bounds prune the last
    // two zones.
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..300);
    assert_eq!(result, SearchResult::at_most(expected));

    // Test IsIn query with NaN and finite values
    let query = SargableQuery::IsIn(vec![
        ScalarValue::Float32(Some(f32::NAN)),
        ScalarValue::Float32(Some(5.0)),
        ScalarValue::Float32(Some(150.0)), // This value exists in the second zone
    ]);
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // Should match all zones since they all contain NaN values
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..500);
    assert_eq!(result, SearchResult::at_most(expected));

    // Test range query that excludes all values
    let query = SargableQuery::Range(
        Bound::Included(ScalarValue::Float32(Some(1000.0))),
        Bound::Included(ScalarValue::Float32(Some(2000.0))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // Every ordinary max is below 1000.0 and a two-sided finite range
    // never admits a NaN, so no zone survives.
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

    // Test IsNull query (should match nothing since there are no null values)
    let query = SargableQuery::IsNull();
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::exact(RowAddrTreeMap::new()));

    // Test range queries with NaN bounds
    // Range with NaN as start bound (included)
    let query = SargableQuery::Range(
        Bound::Included(ScalarValue::Float32(Some(f32::NAN))),
        Bound::Unbounded,
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // Should match all zones since they all contain NaN values
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..500);
    assert_eq!(result, SearchResult::at_most(expected));

    // Range with NaN as end bound (included)
    let query = SargableQuery::Range(
        Bound::Unbounded,
        Bound::Included(ScalarValue::Float32(Some(f32::NAN))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // Should match all zones since they all contain NaN values
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..500);
    assert_eq!(result, SearchResult::at_most(expected));

    // Range with NaN as end bound (excluded)
    let query = SargableQuery::Range(
        Bound::Unbounded,
        Bound::Excluded(ScalarValue::Float32(Some(f32::NAN))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // Should match all zones since everything is less than NaN
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..500);
    assert_eq!(result, SearchResult::at_most(expected));

    // Range with NaN as start bound (excluded)
    let query = SargableQuery::Range(
        Bound::Excluded(ScalarValue::Float32(Some(f32::NAN))),
        Bound::Unbounded,
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // A positive NaN with a larger payload sorts above the canonical NaN
    // bound, and the counts cannot rule one out, so every NaN-bearing zone
    // stays and the predicate recheck decides.
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..500);
    assert_eq!(result, SearchResult::at_most(expected));

    // Test IsIn query with mixed float types (Float16, Float32, Float64)
    let query = SargableQuery::IsIn(vec![
        ScalarValue::Float16(Some(half::f16::NAN)),
        ScalarValue::Float32(Some(f32::NAN)),
        ScalarValue::Float64(Some(f64::NAN)),
        ScalarValue::Float32(Some(5.0)),
    ]);
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // Should match all zones since they all contain NaN values
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..500);
    assert_eq!(result, SearchResult::at_most(expected));
}

#[tokio::test]
// Test data that belongs to the same fragment but coming from different batches
async fn test_basic_zonemap_index() {
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let data = arrow_array::Int32Array::from_iter_values(0..=100);
    let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Int32, false),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let data =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
    let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::once(std::future::ready(Ok(data))),
    ));

    ZoneMapIndexPlugin::train_zonemap_index(
        data_stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(100)),
    )
    .await
    .unwrap();

    log::debug!("Successfully wrote the index file");

    // Read the raw index file back and check its contents
    let index_file = test_store.open_index_file(ZONEMAP_FILENAME).await.unwrap();
    // Print the metadata from the index_file
    let metadata = index_file.schema().metadata.clone();
    let record_batch = index_file
        .read_record_batch(0, index_file.num_rows() as u64)
        .await
        .unwrap();
    assert_eq!(record_batch.num_rows(), 2);
    assert_eq!(
        record_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap()
            .values(),
        &[0, 100]
    );
    assert_eq!(
        record_batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap()
            .values(),
        &[99, 100]
    );
    assert_eq!(
        record_batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .unwrap()
            .values(),
        &[0, 0]
    );
    assert_eq!(metadata.get(ZONEMAP_SIZE_META_KEY).unwrap(), "100");

    // Read the index file back and check its contents
    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .expect("Failed to load ZoneMapIndex");
    assert_eq!(index.zones.len(), 2);
    assert_eq!(
        index.zones,
        vec![
            ZoneMapStatistics {
                min: ScalarValue::Int32(Some(0)),
                max: ScalarValue::Int32(Some(99)),
                null_count: 0,
                nan_count: 0,
                negative_nan_count: Some(0),
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 0,
                    length: 100,
                },
            },
            ZoneMapStatistics {
                min: ScalarValue::Int32(Some(100)),
                max: ScalarValue::Int32(Some(100)),
                null_count: 0,
                nan_count: 0,
                negative_nan_count: Some(0),
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 100,
                    length: 1,
                },
            }
        ]
    );
    // Verify nan_count is 0 for all zones (no NaN values in integer data)
    for (i, zone) in index.zones.iter().enumerate() {
        assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
    }

    assert_eq!(index.data_type, DataType::Int32);
    assert_eq!(index.rows_per_zone, 100);
    assert_eq!(
        index.calculate_included_frags().await.unwrap(),
        RoaringBitmap::from_iter(0..1)
    );

    // Test search functionality

    // 1. Range query: (50, +inf)
    let query = SargableQuery::Range(
        Bound::Excluded(ScalarValue::Int32(Some(50))),
        Bound::Unbounded,
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(0..=100));

    // 2. Range query: [0, 50]
    let query = SargableQuery::Range(
        Bound::Included(ScalarValue::Int32(Some(0))),
        Bound::Included(ScalarValue::Int32(Some(50))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(0..=99));

    // 3. Range query: [101, 200] (should only match the second zone, which is row 100)
    let query = SargableQuery::Range(
        Bound::Included(ScalarValue::Int32(Some(101))),
        Bound::Included(ScalarValue::Int32(Some(200))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // Only row 100 is in the second zone, but its value is 100, so this should be empty
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

    // 4. Range query: [100, 100] (should match only the last row)
    let query = SargableQuery::Range(
        Bound::Included(ScalarValue::Int32(Some(100))),
        Bound::Included(ScalarValue::Int32(Some(100))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(100..=100));

    // 5. Equals query: 0 (should match first row)
    let query = SargableQuery::Equals(ScalarValue::Int32(Some(0)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(0..=99));

    // 6. Equals query: 100 (should match only last row)
    let query = SargableQuery::Equals(ScalarValue::Int32(Some(100)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(100..=100));

    // 7. Equals query: 101 (should match nothing)
    let query = SargableQuery::Equals(ScalarValue::Int32(Some(101)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

    // 8. IsNull query (no nulls in data, should match nothing)
    let query = SargableQuery::IsNull();
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::exact(RowAddrTreeMap::new()));
    // 9. IsIn query: [0, 100, 101, 50]
    let query = SargableQuery::IsIn(vec![
        ScalarValue::Int32(Some(0)),
        ScalarValue::Int32(Some(100)),
        ScalarValue::Int32(Some(101)),
        ScalarValue::Int32(Some(50)),
    ]);
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // 0 and 50 are in the first zone, 100 in the second, 101 is not present
    assert_eq!(result, SearchResult::at_most(0..=100));

    // 10. IsIn query: [101, 102] (should match nothing)
    let query = SargableQuery::IsIn(vec![
        ScalarValue::Int32(Some(101)),
        ScalarValue::Int32(Some(102)),
    ]);
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

    // 11. IsIn query: [null] (should match nothing, as there are no nulls)
    let query = SargableQuery::IsIn(vec![ScalarValue::Int32(None)]);
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

    // 12. Equals query: null (should match nothing, as there are no nulls)
    let query = SargableQuery::Equals(ScalarValue::Int32(None));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));
}

#[tokio::test]
// Test zonemap with same fragment from multiple batches
async fn test_complex_zonemap_index() {
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Create data that will produce the expected zonemap zones
    let data =
        arrow_array::Int64Array::from_iter_values(0..(ROWS_PER_ZONE_DEFAULT * 2 + 42) as i64);
    let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Int64, false),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let data =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
    let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::once(std::future::ready(Ok(data))),
    ));

    ZoneMapIndexPlugin::train_zonemap_index(
        data_stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::default()),
    )
    .await
    .unwrap();

    log::debug!("Successfully wrote the index file");

    // Read the index file back and check its contents
    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .expect("Failed to load ZoneMapIndex");
    assert_eq!(index.zones.len(), 3);
    assert_eq!(
        index.zones,
        vec![
            ZoneMapStatistics {
                min: ScalarValue::Int64(Some(0)),
                max: ScalarValue::Int64(Some(8191)),
                null_count: 0,
                nan_count: 0,
                negative_nan_count: Some(0),
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 0,
                    length: 8192,
                },
            },
            ZoneMapStatistics {
                min: ScalarValue::Int64(Some(8192)),
                max: ScalarValue::Int64(Some(16383)),
                null_count: 0,
                nan_count: 0,
                negative_nan_count: Some(0),
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 8192,
                    length: 8192,
                },
            },
            ZoneMapStatistics {
                min: ScalarValue::Int64(Some(16384)),
                max: ScalarValue::Int64(Some(16425)),
                null_count: 0,
                nan_count: 0,
                negative_nan_count: Some(0),
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 16384,
                    length: 42,
                },
            }
        ]
    );
    // Verify nan_count is 0 for all zones (no NaN values in integer data)
    for (i, zone) in index.zones.iter().enumerate() {
        assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
    }

    assert_eq!(index.data_type, DataType::Int64);
    assert_eq!(index.rows_per_zone, ROWS_PER_ZONE_DEFAULT);
    assert_eq!(
        index.calculate_included_frags().await.unwrap(),
        RoaringBitmap::from_iter(0..1)
    );

    // TODO: Test search functionality
    // Test search functionality

    // Search for a value in the first zone
    let query = SargableQuery::Equals(ScalarValue::Int64(Some(1000)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // Should match row 1000 in fragment 0: row address = (0 << 32) + 1000 = 1000
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..=8191);
    assert_eq!(result, SearchResult::at_most(expected));

    // Search for a value in the second zone
    let query = SargableQuery::Equals(ScalarValue::Int64(Some(9000)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // Should match row 9000 in fragment 0: row address = (0 << 32) + 9000 = 9000
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(8192..=16383);
    assert_eq!(result, SearchResult::at_most(expected));

    // Search for a value not present in any zone
    let query = SargableQuery::Equals(ScalarValue::Int64(Some(20000)));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

    // Search for a range that spans multiple zones
    let query = SargableQuery::Range(
        Bound::Included(ScalarValue::Int64(Some(9000))),
        Bound::Included(ScalarValue::Int64(Some(16400))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    // Should match all rows from 8000 to 16400 (inclusive)
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(8192..=16425);
    assert_eq!(result, SearchResult::at_most(expected));
}

#[tokio::test]
// Test zonemap with multiple fragments from different batches
async fn test_multiple_fragments_zonemap() {
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Int64, false),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));

    // Create multiple fragments with data that will produce expected zones
    // Fragment 0: values 0-8191 (first zone)
    let fragment0_data = arrow_array::Int64Array::from_iter_values(0..ROWS_PER_ZONE_DEFAULT as i64);
    let fragment0_row_ids = UInt64Array::from_iter_values(0..ROWS_PER_ZONE_DEFAULT);
    let fragment0_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(fragment0_data), Arc::new(fragment0_row_ids)],
    )
    .unwrap();

    // Fragment 1: values 8192-16383 (second zone)
    let fragment1_data = arrow_array::Int64Array::from_iter_values(
        (ROWS_PER_ZONE_DEFAULT as i64)..((ROWS_PER_ZONE_DEFAULT * 2) as i64),
    );
    let fragment1_row_ids =
        UInt64Array::from_iter_values((0..ROWS_PER_ZONE_DEFAULT).map(|i| i + (1 << 32)));
    let fragment1_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(fragment1_data), Arc::new(fragment1_row_ids)],
    )
    .unwrap();

    // Fragment 2: values 16384-16426 (third zone)
    let fragment2_data = arrow_array::Int64Array::from_iter_values(
        ((ROWS_PER_ZONE_DEFAULT * 2) as i64)..((ROWS_PER_ZONE_DEFAULT * 2 + 42) as i64),
    );
    let fragment2_row_ids = UInt64Array::from_iter_values((0..42).map(|i| (i as u64) + (2 << 32)));
    let fragment2_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(fragment2_data), Arc::new(fragment2_row_ids)],
    )
    .unwrap();

    // Each fragment is broken into few batches
    {
        // Create a stream with multiple batches (fragments)
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![
                Ok(fragment0_batch.clone()),
                Ok(fragment1_batch.clone()),
                Ok(fragment2_batch.clone()),
            ]),
        ));
        ZoneMapIndexPlugin::train_zonemap_index(
            data_stream,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderParams::new(5000)),
        )
        .await
        .unwrap();

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(
            test_store.clone(),
            None,
            &LanceCache::no_cache(),
            false,
            ZoneMapIndexVersion::V1,
        )
        .await
        .expect("Failed to load ZoneMapIndex");
        assert_eq!(index.zones.len(), 5);
        assert_eq!(
            index.zones,
            vec![
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(0)),
                    max: ScalarValue::Int64(Some(4999)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 0,
                        length: 5000,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(5000)),
                    max: ScalarValue::Int64(Some(8191)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 5000,
                        length: 3192,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(8192)),
                    max: ScalarValue::Int64(Some(13191)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 1,
                        start: 0,
                        length: 5000,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(13192)),
                    max: ScalarValue::Int64(Some(16383)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 1,
                        start: 5000,
                        length: 3192,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(16384)),
                    max: ScalarValue::Int64(Some(16425)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 2,
                        start: 0,
                        length: 42,
                    },
                }
            ]
        );
        // Verify nan_count is 0 for all zones (no NaN values in integer data)
        for (i, zone) in index.zones.iter().enumerate() {
            assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
        }

        assert_eq!(index.data_type, DataType::Int64);
        assert_eq!(index.rows_per_zone, 5000);
        assert_eq!(
            index.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter(0..3)
        );

        // Verify _rowaddr column values are properly assigned
        let verify_data_stream: SendableRecordBatchStream =
            Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                stream::iter(vec![
                    Ok(fragment0_batch.clone()),
                    Ok(fragment1_batch.clone()),
                    Ok(fragment2_batch.clone()),
                ]),
            ));
        let batches: Vec<RecordBatch> = verify_data_stream.try_collect().await.unwrap();

        assert_eq!(batches.len(), 3);

        // Check fragment 0 _rowaddr values (should start from 0)
        let fragment0_rowaddr_col = batches[0].column_by_name(ROW_ADDR).unwrap();
        let fragment0_rowaddrs = fragment0_rowaddr_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(
            fragment0_rowaddrs.values().len(),
            ROWS_PER_ZONE_DEFAULT as usize
        );
        assert_eq!(fragment0_rowaddrs.values()[0], 0);
        assert_eq!(
            fragment0_rowaddrs.values()[fragment0_rowaddrs.values().len() - 1],
            8191
        );

        // Check fragment 1 _rowaddr values (should start from fragment_id=1)
        let fragment1_rowaddr_col = batches[1].column_by_name(ROW_ADDR).unwrap();
        let fragment1_rowaddrs = fragment1_rowaddr_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(
            fragment1_rowaddrs.values().len(),
            ROWS_PER_ZONE_DEFAULT as usize
        );
        assert_eq!(fragment1_rowaddrs.values()[0], 1u64 << 32); // fragment_id=1, local_offset=0
        assert_eq!(
            fragment1_rowaddrs.values()[fragment1_rowaddrs.values().len() - 1],
            8191 | (1u64 << 32)
        );

        // Check fragment 2 _rowaddr values (should start from fragment_id=2)
        let fragment2_rowaddr_col = batches[2].column_by_name(ROW_ADDR).unwrap();
        let fragment2_rowaddrs = fragment2_rowaddr_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(fragment2_rowaddrs.values().len(), 42);
        assert_eq!(fragment2_rowaddrs.values()[0], 2u64 << 32); // fragment_id=2, local_offset=0
        assert_eq!(
            fragment2_rowaddrs.values()[fragment2_rowaddrs.values().len() - 1],
            (2u64 << 32) | 41
        );

        // Add a few tests for search functionality

        // Test range query that spans multiple fragments
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int64(Some(5000))),
            Bound::Included(ScalarValue::Int64(Some(12000))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should include zones from fragments 0 and 1 since they overlap with range 5000-12000
        let mut expected = RowAddrTreeMap::new();
        // zone 1
        expected.insert_range(5000..8192);
        // zone 2
        expected.insert_range((1u64 << 32)..((1u64 << 32) + 5000));
        assert_eq!(result, SearchResult::at_most(expected));

        // Test exact match query from zone 2
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(8192)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should include zone 2 since it contains value 8192
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range((1u64 << 32)..((1u64 << 32) + 5000));
        assert_eq!(result, SearchResult::at_most(expected));

        // Test exact match query from zone 4
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(16385)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should include zone 4 since it contains value 16385
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(2u64 << 32..((2u64 << 32) + 42));
        assert_eq!(result, SearchResult::at_most(expected));

        // Test query that matches nothing
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(99999)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

        // Test is_in query
        let query = SargableQuery::IsIn(vec![ScalarValue::Int64(Some(16385))]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(2u64 << 32..((2u64 << 32) + 42));
        assert_eq!(result, SearchResult::at_most(expected));

        // Test equals query with null
        let query = SargableQuery::Equals(ScalarValue::Int64(None));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..=16425);
        // expected = {:?}", expected
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));
    }

    //  Each fragment is its own batch
    {
        // Create a stream with multiple batches (fragments)
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![
                Ok(fragment0_batch.clone()),
                Ok(fragment1_batch.clone()),
                Ok(fragment2_batch.clone()),
            ]),
        ));
        ZoneMapIndexPlugin::train_zonemap_index(
            data_stream,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderParams::default()),
        )
        .await
        .unwrap();

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(
            test_store.clone(),
            None,
            &LanceCache::no_cache(),
            false,
            ZoneMapIndexVersion::V1,
        )
        .await
        .expect("Failed to load ZoneMapIndex");
        assert_eq!(index.zones.len(), 3);
        assert_eq!(
            index.zones,
            vec![
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(0)),
                    max: ScalarValue::Int64(Some(8191)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 0,
                        length: 8192,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(8192)),
                    max: ScalarValue::Int64(Some(16383)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 1,
                        start: 0,
                        length: 8192,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(16384)),
                    max: ScalarValue::Int64(Some(16425)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 2,
                        start: 0,
                        length: 42,
                    },
                }
            ]
        );
        // Verify nan_count is 0 for all zones (no NaN values in integer data)
        for (i, zone) in index.zones.iter().enumerate() {
            assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
        }

        assert_eq!(index.data_type, DataType::Int64);
        assert_eq!(index.rows_per_zone, ROWS_PER_ZONE_DEFAULT);
        assert_eq!(
            index.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter(0..3)
        );
    }

    //  All fragments are in the same batch
    {
        // Create a stream with multiple batches (fragments)
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![
                Ok(fragment0_batch.clone()),
                Ok(fragment1_batch.clone()),
                Ok(fragment2_batch.clone()),
            ]),
        ));
        ZoneMapIndexPlugin::train_zonemap_index(
            data_stream,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderParams::new(ROWS_PER_ZONE_DEFAULT * 3)),
        )
        .await
        .unwrap();

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(
            test_store.clone(),
            None,
            &LanceCache::no_cache(),
            false,
            ZoneMapIndexVersion::V1,
        )
        .await
        .expect("Failed to load ZoneMapIndex");
        assert_eq!(index.zones.len(), 3);
        assert_eq!(
            index.zones,
            vec![
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(0)),
                    max: ScalarValue::Int64(Some(8191)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 0,
                        length: 8192,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(8192)),
                    max: ScalarValue::Int64(Some(16383)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 1,
                        start: 0,
                        length: 8192,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(16384)),
                    max: ScalarValue::Int64(Some(16425)),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: ZoneBound {
                        fragment_id: 2,
                        start: 0,
                        length: 42,
                    },
                }
            ]
        );
        // Verify nan_count is 0 for all zones (no NaN values in integer data)
        for (i, zone) in index.zones.iter().enumerate() {
            assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
        }

        assert_eq!(index.data_type, DataType::Int64);
        assert_eq!(index.rows_per_zone, ROWS_PER_ZONE_DEFAULT * 3);
    }
}

#[tokio::test]
async fn test_fragment_id_assignment() {
    // Test that fragment IDs are properly assigned in _rowaddr values
    let schema = Arc::new(Schema::new(vec![Field::new(
        VALUE_COLUMN_NAME,
        DataType::Int32,
        false,
    )]));

    // Create multiple fragments
    let fragment0_data = arrow_array::Int32Array::from_iter_values(0..5);
    let fragment0_batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(fragment0_data)]).unwrap();

    let fragment1_data = arrow_array::Int32Array::from_iter_values(5..10);
    let fragment1_batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(fragment1_data)]).unwrap();

    let aligned_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::iter(vec![Ok(fragment0_batch), Ok(fragment1_batch)]),
    ));

    let aligned_stream = add_row_addr(aligned_stream);

    let batches: Vec<RecordBatch> = aligned_stream.try_collect().await.unwrap();

    assert_eq!(batches.len(), 2);

    // Check fragment 0 _rowaddr values
    let fragment0_rowaddr_col = batches[0].column_by_name(ROW_ADDR).unwrap();
    let fragment0_rowaddrs = fragment0_rowaddr_col
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();

    // Fragment 0 should have _rowaddr values: 0, 1, 2, 3, 4
    assert_eq!(fragment0_rowaddrs.values(), &[0, 1, 2, 3, 4]);

    // Check fragment 1 _rowaddr values
    let fragment1_rowaddr_col = batches[1].column_by_name(ROW_ADDR).unwrap();
    let fragment1_rowaddrs = fragment1_rowaddr_col
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();

    // Fragment 1 should have _rowaddr values: (1 << 32) | 0, (1 << 32) | 1, etc.
    // which is: 4294967296, 4294967297, 4294967298, 4294967299, 4294967300
    assert_eq!(
        fragment1_rowaddrs.values(),
        &[4294967296, 4294967297, 4294967298, 4294967299, 4294967300]
    );
}

#[tokio::test]
async fn test_like_prefix_query() {
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Create zones with different string ranges
    // Zone 0: ["aaa", "azz"] - should NOT match "foo%"
    // Zone 1: ["bar", "baz"] - should NOT match "foo%"
    // Zone 2: ["fa", "foz"]  - should match "foo%" (contains potential matches)
    // Zone 3: ["fop", "fzz"] - should NOT match "foo%" (all values >= "fop")
    // Zone 4: ["foo", "foobar"] - should match "foo%"
    // Zone 5: ["gaa", "gzz"] - should NOT match "foo%"

    let zones = vec![
        ZoneMapStatistics {
            min: ScalarValue::Utf8(Some("aaa".to_string())),
            max: ScalarValue::Utf8(Some("azz".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 100,
            },
        },
        ZoneMapStatistics {
            min: ScalarValue::Utf8(Some("bar".to_string())),
            max: ScalarValue::Utf8(Some("baz".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 1,
                start: 0,
                length: 100,
            },
        },
        ZoneMapStatistics {
            min: ScalarValue::Utf8(Some("fa".to_string())),
            max: ScalarValue::Utf8(Some("foz".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 2,
                start: 0,
                length: 100,
            },
        },
        ZoneMapStatistics {
            min: ScalarValue::Utf8(Some("fop".to_string())),
            max: ScalarValue::Utf8(Some("fzz".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 3,
                start: 0,
                length: 100,
            },
        },
        ZoneMapStatistics {
            min: ScalarValue::Utf8(Some("foo".to_string())),
            max: ScalarValue::Utf8(Some("foobar".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 4,
                start: 0,
                length: 100,
            },
        },
        ZoneMapStatistics {
            min: ScalarValue::Utf8(Some("gaa".to_string())),
            max: ScalarValue::Utf8(Some("gzz".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 5,
                start: 0,
                length: 100,
            },
        },
    ];

    let index = ZoneMapIndex {
        zones,
        data_type: DataType::Utf8,
        rows_per_zone: ROWS_PER_ZONE_DEFAULT,
        use_seeds: false,
        version: ZoneMapIndexVersion::V1,
        store_null_bitmap: true,
        store: test_store,
        fri: None,
        batch_remapper: None,
        index_cache: WeakLanceCache::from(&LanceCache::no_cache()),
        null_rows: None,
    };

    // Test LikePrefix query for "foo"
    let query = SargableQuery::LikePrefix(ScalarValue::Utf8(Some("foo".to_string())));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // Should match zones 2 and 4 only
    let mut expected = RowAddrTreeMap::new();
    // Zone 2: fragment 2
    expected.insert_range((2u64 << 32)..((2u64 << 32) + 100));
    // Zone 4: fragment 4
    expected.insert_range((4u64 << 32)..((4u64 << 32) + 100));

    assert_eq!(result, SearchResult::at_most(expected));
}

#[tokio::test]
async fn test_like_prefix_edge_cases() {
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Test edge cases for LIKE prefix
    let zones = vec![
        // Zone with values that contain the prefix exactly
        ZoneMapStatistics {
            min: ScalarValue::Utf8(Some("test".to_string())),
            max: ScalarValue::Utf8(Some("test".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 100,
            },
        },
        // Zone with values that span across the prefix boundary
        ZoneMapStatistics {
            min: ScalarValue::Utf8(Some("te".to_string())),
            max: ScalarValue::Utf8(Some("tf".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 1,
                start: 0,
                length: 100,
            },
        },
        // Zone completely before prefix
        ZoneMapStatistics {
            min: ScalarValue::Utf8(Some("abc".to_string())),
            max: ScalarValue::Utf8(Some("def".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 2,
                start: 0,
                length: 100,
            },
        },
    ];

    let index = ZoneMapIndex {
        zones,
        data_type: DataType::Utf8,
        rows_per_zone: ROWS_PER_ZONE_DEFAULT,
        use_seeds: false,
        version: ZoneMapIndexVersion::V1,
        store_null_bitmap: true,
        store: test_store,
        fri: None,
        batch_remapper: None,
        index_cache: WeakLanceCache::from(&LanceCache::no_cache()),
        null_rows: None,
    };

    // Test LikePrefix "test"
    let query = SargableQuery::LikePrefix(ScalarValue::Utf8(Some("test".to_string())));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // Should match zones 0 and 1
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..100); // Zone 0: fragment 0
    expected.insert_range((1u64 << 32)..((1u64 << 32) + 100));

    assert_eq!(result, SearchResult::at_most(expected));

    // Test empty prefix - should match all zones
    let query = SargableQuery::LikePrefix(ScalarValue::Utf8(Some("".to_string())));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    let mut expected = RowAddrTreeMap::new();
    expected.insert_range(0..100); // Zone 0: fragment 0
    expected.insert_range((1u64 << 32)..((1u64 << 32) + 100));
    expected.insert_range((2u64 << 32)..((2u64 << 32) + 100));

    assert_eq!(result, SearchResult::at_most(expected));
}

#[tokio::test]
async fn test_like_prefix_large_utf8() {
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Test with LargeUtf8 type
    let zones = vec![
        ZoneMapStatistics {
            min: ScalarValue::LargeUtf8(Some("aaa".to_string())),
            max: ScalarValue::LargeUtf8(Some("azz".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 100,
            },
        },
        ZoneMapStatistics {
            min: ScalarValue::LargeUtf8(Some("foo".to_string())),
            max: ScalarValue::LargeUtf8(Some("foobar".to_string())),
            null_count: 0,
            nan_count: 0,
            negative_nan_count: Some(0),
            bound: ZoneBound {
                fragment_id: 1,
                start: 0,
                length: 100,
            },
        },
    ];

    let index = ZoneMapIndex {
        zones,
        data_type: DataType::LargeUtf8,
        rows_per_zone: ROWS_PER_ZONE_DEFAULT,
        use_seeds: false,
        version: ZoneMapIndexVersion::V1,
        store_null_bitmap: true,
        store: test_store,
        fri: None,
        batch_remapper: None,
        index_cache: WeakLanceCache::from(&LanceCache::no_cache()),
        null_rows: None,
    };

    // Test LikePrefix with LargeUtf8
    let query = SargableQuery::LikePrefix(ScalarValue::LargeUtf8(Some("foo".to_string())));
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

    // Should match only zone 1
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range((1u64 << 32)..((1u64 << 32) + 100));

    assert_eq!(result, SearchResult::at_most(expected));
}

#[test]
fn test_compute_next_prefix() {
    use super::compute_next_prefix;

    // Basic cases
    assert_eq!(compute_next_prefix("foo"), Some("fop".to_string()));
    assert_eq!(compute_next_prefix("abc"), Some("abd".to_string()));
    assert_eq!(compute_next_prefix("a"), Some("b".to_string()));
    assert_eq!(compute_next_prefix("z"), Some("{".to_string())); // 'z' + 1 = '{'

    // Edge case: prefix with 'z' at the end
    assert_eq!(compute_next_prefix("abz"), Some("ab{".to_string()));

    // Edge case with tilde (~) which is 0x7E
    assert_eq!(compute_next_prefix("ab~"), Some("ab\x7f".to_string()));

    // Empty prefix
    assert_eq!(compute_next_prefix(""), None);

    // Non-ASCII: works correctly by incrementing Unicode code points
    // é (U+00E9) -> ê (U+00EA)
    assert_eq!(compute_next_prefix("café"), Some("cafê".to_string()));
    // 中 (U+4E2D) -> 丮 (U+4E2E)
    assert_eq!(compute_next_prefix("abc中"), Some("abc丮".to_string()));
    // ÿ (U+00FF) -> Ā (U+0100) - crosses byte boundary but works
    assert_eq!(compute_next_prefix("cafÿ"), Some("cafĀ".to_string()));

    // Edge case: character just before surrogate range
    // U+D7FF -> U+E000 (skips surrogate range U+D800-U+DFFF)
    assert_eq!(
        compute_next_prefix("a\u{D7FF}"),
        Some("a\u{E000}".to_string())
    );

    // Edge case: max Unicode character U+10FFFF, falls back to previous char
    assert_eq!(compute_next_prefix("ab\u{10FFFF}"), Some("ac".to_string()));
    // All max characters
    assert_eq!(compute_next_prefix("\u{10FFFF}\u{10FFFF}"), None);
}

// When merging zone map segments, if ANY source segment has null_rows = None
// (legacy — null positions unknown), the merged result must also be None.
// The bug: any_null_bitmap is set to true as soon as one source has Some(...),
// and the None sources are silently skipped.  The merged index then has
// null_rows = Some(partial_bitmap), so an IsNull search returns exact results
// that only cover the modern segment's nulls — a false negative for the legacy
// segment whose null positions were never tracked.
#[tokio::test]
async fn test_merge_with_legacy_none_segment_not_treated_as_no_nulls() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    use arrow_array::{Int32Array, UInt32Array};

    // Index A: fragment 0, modern — has a complete null bitmap with 2 known null rows.
    let schema_a = Arc::new(Schema::new(vec![
        Field::new("min", DataType::Int32, true),
        Field::new("max", DataType::Int32, true),
        Field::new("null_count", DataType::UInt32, false),
        Field::new("nan_count", DataType::UInt32, false),
        Field::new("negative_nan_count", DataType::UInt32, false),
        Field::new("fragment_id", DataType::UInt64, false),
        Field::new("zone_start", DataType::UInt64, false),
        Field::new("zone_length", DataType::UInt64, false),
    ]));
    let batch_a = RecordBatch::try_new(
        schema_a,
        vec![
            Arc::new(Int32Array::from(vec![Some(1i32)])) as _,
            Arc::new(Int32Array::from(vec![Some(5i32)])) as _,
            Arc::new(UInt32Array::from(vec![2u32])) as _,
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![10u64])) as _,
        ],
    )
    .unwrap();
    let mut modern_null_rows = RowAddrTreeMap::new();
    modern_null_rows.insert(3); // frag 0 row 3
    modern_null_rows.insert(7); // frag 0 row 7
    let cache = LanceCache::no_cache();
    let index_a = Arc::new(
        ZoneMapIndex::try_from_serialized(
            batch_a,
            store.clone(),
            None,
            &cache,
            10,
            Some(modern_null_rows), // modern: complete bitmap
            false,
            true,
            ZoneMapIndexVersion::V1,
        )
        .unwrap(),
    );

    // Index B: fragment 1, legacy — null_rows = None despite null_count = 3.
    let schema_b = Arc::new(Schema::new(vec![
        Field::new("min", DataType::Int32, true),
        Field::new("max", DataType::Int32, true),
        Field::new("null_count", DataType::UInt32, false),
        Field::new("nan_count", DataType::UInt32, false),
        Field::new("negative_nan_count", DataType::UInt32, false),
        Field::new("fragment_id", DataType::UInt64, false),
        Field::new("zone_start", DataType::UInt64, false),
        Field::new("zone_length", DataType::UInt64, false),
    ]));
    let batch_b = RecordBatch::try_new(
        schema_b,
        vec![
            Arc::new(Int32Array::from(vec![Some(10i32)])) as _,
            Arc::new(Int32Array::from(vec![Some(20i32)])) as _,
            Arc::new(UInt32Array::from(vec![3u32])) as _,
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt64Array::from(vec![1u64])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![10u64])) as _,
        ],
    )
    .unwrap();
    let index_b = Arc::new(
        ZoneMapIndex::try_from_serialized(
            batch_b,
            store.clone(),
            None,
            &cache,
            10,
            None, // legacy: null positions unknown
            false,
            true,
            ZoneMapIndexVersion::V1,
        )
        .unwrap(),
    );

    let dest_tmpdir = TempObjDir::default();
    let dest_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        dest_tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let all_frags = RoaringBitmap::from_iter([0u32, 1]);
    merge_zonemap_indices(
        &[index_a.as_ref(), index_b.as_ref()],
        dest_store.as_ref(),
        &all_frags,
    )
    .await
    .unwrap();

    let merged = ZoneMapIndex::load(
        dest_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    // Index B had null_rows = None, so the merged index cannot know all null positions.
    // IsNull must NOT return exact — that would be a false negative for fragment 1's nulls.
    let result = merged
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();

    // With the bug: any_null_bitmap=true (from A) → null_rows=Some(A's bitmap only)
    //              → IsNull returns exact, missing B's unknown nulls ← FALSE NEGATIVE
    // With the fix: any_null_bitmap=false (because B is None) → null_rows=None
    //              → IsNull falls through to zone scan → AtMost
    assert!(
        !result.is_exact(),
        "IsNull on a merged index where one source had null_rows=None must not return \
         exact; the legacy segment had null_count=3 so its nulls exist at unknown positions"
    );
}

// Writes a zonemap file in the legacy format (no null bitmap global buffer),
// simulating an index created before the null bitmap feature was added.
async fn write_legacy_zonemap(store: &dyn IndexStore, null_count: u32) {
    use arrow_array::{Int32Array, UInt32Array};
    let schema = Arc::new(Schema::new(vec![
        Field::new("min", DataType::Int32, true),
        Field::new("max", DataType::Int32, true),
        Field::new("null_count", DataType::UInt32, false),
        Field::new("nan_count", DataType::UInt32, false),
        Field::new("fragment_id", DataType::UInt64, false),
        Field::new("zone_start", DataType::UInt64, false),
        Field::new("zone_length", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![Some(0)])) as _,
            Arc::new(Int32Array::from(vec![Some(99)])) as _,
            Arc::new(UInt32Array::from(vec![null_count])) as _,
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![100u64])) as _,
        ],
    )
    .unwrap();
    let mut file_schema = schema.as_ref().clone();
    file_schema
        .metadata
        .insert(ZONEMAP_SIZE_META_KEY.to_string(), "8192".to_string());
    let mut writer = store
        .new_index_file(ZONEMAP_FILENAME, Arc::new(file_schema))
        .await
        .unwrap();
    writer.write_record_batch(batch).await.unwrap();
    writer.finish().await.unwrap();
}

#[tokio::test]
async fn test_legacy_zonemap_no_null_bitmap() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Write a legacy index with one zone that has nulls but no null bitmap.
    write_legacy_zonemap(store.as_ref(), 10).await;

    let index = ZoneMapIndex::load(
        store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V0,
    )
    .await
    .expect("failed to load legacy zonemap");

    assert!(
        index.null_rows.is_none(),
        "legacy index should have no null bitmap"
    );

    // IS NULL should fall back to the zone-scan path and return AtMost, not Exact.
    let result = index
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        !result.is_exact(),
        "IS NULL on a legacy index should not be exact"
    );
}

#[test]
fn test_zone_map_seed_writer_null_runs_match_positions() {
    use crate::scalar::seed::IndexSeedWriter;

    // Validity patterns: scattered, a long run, leading and trailing nulls.
    let values: Vec<Option<i32>> = (0..1000)
        .map(|i| {
            if i < 7 || (200..600).contains(&i) || i % 13 == 0 || i >= 990 {
                None
            } else {
                Some(i)
            }
        })
        .collect();
    let expected: RoaringBitmap = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_none())
        .map(|(i, _)| i as u32)
        .collect();
    let array: ArrayRef = Arc::new(Int32Array::from(values));

    // Observed as one array, as slices that split the runs, and with
    // zones that split the runs: the recorded positions are identical.
    let mut whole = ZoneMapSeedWriter::new("c", 64, DataType::Int32).unwrap();
    whole.observe_batch(&array).unwrap();
    let mut sliced = ZoneMapSeedWriter::new("c", 1000, DataType::Int32).unwrap();
    for (start, len) in [(0, 3), (3, 300), (303, 500), (803, 197)] {
        sliced.observe_batch(&array.slice(start, len)).unwrap();
    }
    for writer in [&mut whole, &mut sliced] {
        let bytes = writer.finish().unwrap().unwrap();
        let (_, bitmap) = ZoneMapSeedWriter::deserialize_seed(0, &bytes).unwrap();
        assert_eq!(bitmap, Some(expected.clone()));
    }

    // No validity buffer means no nulls: the run scan records nothing.
    let valid: ArrayRef = Arc::new(Int32Array::from_iter_values(0..10));
    let mut writer = ZoneMapSeedWriter::new("c", 64, DataType::Int32).unwrap();
    writer.record_null_runs(&valid, 0);
    assert!(writer.null_offsets.is_empty());
}

#[tokio::test]
async fn test_zone_map_seed_writer_round_trip() {
    use crate::scalar::seed::IndexSeedWriter;
    use datafusion_common::ScalarValue;

    let rows_per_zone = 4u64;
    let data_type = DataType::Int32;
    let mut writer = ZoneMapSeedWriter::new("test_col", rows_per_zone, data_type).unwrap();

    // Batch 1: values 0..4 (fills exactly one zone)
    let batch1: ArrayRef = Arc::new(Int32Array::from_iter_values(0..4));
    writer.observe_batch(&batch1).unwrap();

    // Batch 2: values 10..14 (fills a second zone exactly)
    let batch2: ArrayRef = Arc::new(Int32Array::from_iter_values(10..14));
    writer.observe_batch(&batch2).unwrap();

    // Batch 3: values 20..22 (partial final zone)
    let batch3: ArrayRef = Arc::new(Int32Array::from_iter_values(20..22));
    writer.observe_batch(&batch3).unwrap();

    let bytes = writer.finish().unwrap().expect("should produce bytes");

    // Check schema metadata key/value format
    assert_eq!(writer.schema_metadata_key(), "lance.seed.test_col");
    let meta_val = writer.schema_metadata_value(3);
    assert_eq!(meta_val, "3:4");

    // Deserialize and verify
    let (zones, null_bitmap) = ZoneMapSeedWriter::deserialize_seed(42, &bytes).unwrap();
    assert_eq!(zones.len(), 3, "expected 3 zones");

    // No nulls in the input, so null bitmap is empty
    assert_eq!(null_bitmap, Some(RoaringBitmap::default()));

    // Zone 0: values 0..4 -> min=0, max=3
    assert_eq!(zones[0].bound.fragment_id, 42);
    assert_eq!(zones[0].bound.start, 0);
    assert_eq!(zones[0].min, ScalarValue::Int32(Some(0)));
    assert_eq!(zones[0].max, ScalarValue::Int32(Some(3)));
    assert_eq!(zones[0].null_count, 0);

    // Zone 1: values 10..14 -> min=10, max=13
    assert_eq!(zones[1].bound.start, 4);
    assert_eq!(zones[1].min, ScalarValue::Int32(Some(10)));
    assert_eq!(zones[1].max, ScalarValue::Int32(Some(13)));

    // Zone 2: values 20..22 -> min=20, max=21, partial zone of 2 rows
    assert_eq!(zones[2].bound.start, 8);
    assert_eq!(
        zones[2].bound.length, 2,
        "partial zone length must be exact"
    );
    assert_eq!(zones[2].min, ScalarValue::Int32(Some(20)));
    assert_eq!(zones[2].max, ScalarValue::Int32(Some(21)));

    // Full zones must have the full rows_per_zone length
    assert_eq!(zones[0].bound.length, rows_per_zone as usize);
    assert_eq!(zones[1].bound.length, rows_per_zone as usize);
}

#[tokio::test]
async fn test_zone_map_seed_writer_spanning_batches() {
    use crate::scalar::seed::IndexSeedWriter;
    use datafusion_common::ScalarValue;

    let rows_per_zone = 5u64;
    let data_type = DataType::Int32;
    let mut writer = ZoneMapSeedWriter::new("val", rows_per_zone, data_type).unwrap();

    // Single batch with 12 values -> should produce 2 complete zones + 1 partial
    let batch: ArrayRef = Arc::new(Int32Array::from_iter_values(0..12));
    writer.observe_batch(&batch).unwrap();

    let bytes = writer.finish().unwrap().expect("should produce bytes");
    let (zones, _null_bitmap) = ZoneMapSeedWriter::deserialize_seed(1, &bytes).unwrap();
    assert_eq!(
        zones.len(),
        3,
        "expected 3 zones from 12 rows with zone size 5"
    );

    // Zone 0: rows 0..5
    assert_eq!(zones[0].bound.length, 5);
    assert_eq!(zones[0].min, ScalarValue::Int32(Some(0)));
    assert_eq!(zones[0].max, ScalarValue::Int32(Some(4)));
    // Zone 1: rows 5..10
    assert_eq!(zones[1].bound.length, 5);
    assert_eq!(zones[1].min, ScalarValue::Int32(Some(5)));
    assert_eq!(zones[1].max, ScalarValue::Int32(Some(9)));
    // Zone 2: rows 10..12 (partial)
    assert_eq!(zones[2].bound.length, 2, "partial zone length must be 2");
    assert_eq!(zones[2].min, ScalarValue::Int32(Some(10)));
    assert_eq!(zones[2].max, ScalarValue::Int32(Some(11)));
}

#[tokio::test]
async fn test_zone_map_seed_writer_empty() {
    use crate::scalar::seed::IndexSeedWriter;

    let mut writer = ZoneMapSeedWriter::new("col", 8, DataType::Int32).unwrap();
    let result = writer.finish().unwrap();
    assert!(result.is_none(), "empty fragment should return None");
}

/// Seeds produced for fragments that contain null values must carry a null
/// bitmap so that `try_update_with_seeds` can reconstruct `null_rows` without
/// rescanning data.
#[tokio::test]
async fn test_seed_null_bitmap_round_trip() {
    use crate::scalar::seed::IndexSeedWriter;
    use arrow_array::Int32Array;
    use lance_select::RowSetOps;

    // rows_per_zone = 4, 10 rows: nulls at positions 1, 3, 5, 9
    //   zone 0: rows 0-3 → nulls at 1, 3
    //   zone 1: rows 4-7 → null at 5
    //   zone 2: rows 8-9 → null at 9
    let rows_per_zone = 4u64;
    let values: ArrayRef = Arc::new(Int32Array::from(vec![
        Some(0),
        None, // offset 1
        Some(2),
        None, // offset 3
        Some(4),
        None, // offset 5
        Some(6),
        Some(7),
        Some(8),
        None, // offset 9
    ]));

    let mut writer = ZoneMapSeedWriter::new("col", rows_per_zone, DataType::Int32).unwrap();
    writer.observe_batch(&values).unwrap();
    let bytes = writer.finish().unwrap().expect("should produce bytes");

    let (zones, null_bitmap) = ZoneMapSeedWriter::deserialize_seed(7, &bytes).unwrap();
    assert_eq!(zones.len(), 3);
    assert_eq!(zones[0].null_count, 2);
    assert_eq!(zones[1].null_count, 1);
    assert_eq!(zones[2].null_count, 1);

    let bitmap = null_bitmap.expect("null bitmap must be present");
    assert!(bitmap.contains(1), "offset 1 must be in bitmap");
    assert!(bitmap.contains(3), "offset 3 must be in bitmap");
    assert!(bitmap.contains(5), "offset 5 must be in bitmap");
    assert!(bitmap.contains(9), "offset 9 must be in bitmap");
    assert_eq!(bitmap.len(), 4, "exactly 4 null positions");

    // Reconstruct a RowAddrTreeMap as try_update_with_seeds would do.
    let mut null_rows = RowAddrTreeMap::new();
    null_rows.insert_bitmap(7, bitmap);
    // Verify the row addresses are (fragment 7 << 32 | offset).
    let frag7 = 7u64 << 32;
    assert!(null_rows.contains(frag7 | 1));
    assert!(null_rows.contains(frag7 | 3));
    assert!(null_rows.contains(frag7 | 5));
    assert!(null_rows.contains(frag7 | 9));
    assert!(!null_rows.contains(frag7));
}

/// After updating a zone map index via seeds from a fragment that contains
/// nulls, IS NULL must return an exact result (not a zone-scan approximation).
#[tokio::test]
async fn test_update_with_seeds_propagates_null_bitmap() {
    use crate::scalar::seed::{FragmentSeed, IndexSeedWriter};
    use crate::scalar::zonemap::{ZoneMapIndexBuilder, ZoneMapSeedWriter};
    use arrow_array::Int32Array;
    use lance_select::RowSetOps;

    // Build an initial zone map index with one fragment that has nulls at rows 0 and 2.
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Initial training: fragment 0, rows 0-3, nulls at 0 and 2.
    let rows_per_zone = 4u64;
    let initial_values: ArrayRef = Arc::new(Int32Array::from(vec![
        None, // frag 0, row 0
        Some(1),
        None, // frag 0, row 2
        Some(3),
    ]));
    let mut builder = ZoneMapIndexBuilder::try_new(
        ZoneMapIndexBuilderParams::new(rows_per_zone),
        DataType::Int32,
    )
    .unwrap();
    let row_addrs: ArrayRef = Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3]));
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        Field::new("value", DataType::Int32, true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![initial_values, row_addrs]).unwrap();
    let stream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::once(async { Ok(batch) }),
    ));
    builder.train(stream).await.unwrap();
    builder.write_index(store.as_ref()).await.unwrap();

    // Load the trained index.
    let index = ZoneMapIndex::load(
        store.clone(),
        None,
        &LanceCache::no_cache(),
        true,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert!(
        index.null_rows.is_some(),
        "initial index must have null bitmap"
    );

    // Build a seed for fragment 1: rows 0-3, null at row 1.
    let seed_values: ArrayRef = Arc::new(Int32Array::from(vec![
        Some(10),
        None, // frag 1, row 1
        Some(12),
        Some(13),
    ]));
    let mut seed_writer = ZoneMapSeedWriter::new("value", rows_per_zone, DataType::Int32).unwrap();
    seed_writer.observe_batch(&seed_values).unwrap();
    let seed_bytes = seed_writer
        .finish()
        .unwrap()
        .expect("seed must produce bytes");

    let seed = FragmentSeed {
        fragment_id: 1,
        physical_rows: 4,
        bytes: seed_bytes,
        metadata_value: format!("0:{rows_per_zone}"),
    };

    // Update the index with the seed.
    let dest_tmpdir = TempObjDir::default();
    let dest_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        dest_tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    index
        .try_update_with_seeds(&[seed], dest_store.as_ref())
        .await
        .unwrap()
        .expect("update must produce a result");

    let updated_index = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        true,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    // The updated index must have a null bitmap covering both fragments.
    let null_rows = updated_index
        .null_rows
        .as_ref()
        .expect("updated index must have null bitmap");

    // Fragment 0: nulls at offsets 0 and 2
    assert!(null_rows.contains(0u64), "frag 0 row 0 must be null");
    assert!(null_rows.contains(2u64), "frag 0 row 2 must be null");
    assert!(!null_rows.contains(1u64), "frag 0 row 1 must not be null");

    // Fragment 1: null at offset 1
    let frag1 = 1u64 << 32;
    assert!(null_rows.contains(frag1 | 1), "frag 1 row 1 must be null");
    assert!(!null_rows.contains(frag1), "frag 1 row 0 must not be null");

    // IS NULL must return exact results (not a zone-scan approximation).
    let result = updated_index
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(result.is_exact(), "IS NULL after seed update must be exact");
}

/// A legacy index (null_rows = None) updated with a seed must still have
/// null_rows = None in the result.  The seed cannot retroactively supply null
/// positions for the rows already in the old index, so the updated index must
/// stay conservative and fall back to zone-scan for IS NULL.
#[tokio::test]
async fn test_update_with_seeds_legacy_index_keeps_null_rows_none() {
    use crate::scalar::seed::{FragmentSeed, IndexSeedWriter};
    use crate::scalar::zonemap::{ZoneMapIndex, ZoneMapSeedWriter};
    use arrow_array::{Int32Array, UInt32Array};

    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Build a legacy index (fragment 0, null_rows = None).
    let schema = Arc::new(Schema::new(vec![
        Field::new("min", DataType::Int32, true),
        Field::new("max", DataType::Int32, true),
        Field::new("null_count", DataType::UInt32, false),
        Field::new("nan_count", DataType::UInt32, false),
        Field::new("negative_nan_count", DataType::UInt32, false),
        Field::new("fragment_id", DataType::UInt64, false),
        Field::new("zone_start", DataType::UInt64, false),
        Field::new("zone_length", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![Some(0i32)])) as _,
            Arc::new(Int32Array::from(vec![Some(9i32)])) as _,
            Arc::new(UInt32Array::from(vec![2u32])) as _, // 2 nulls, positions unknown
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![10u64])) as _,
        ],
    )
    .unwrap();
    let cache = LanceCache::no_cache();
    let legacy_index = Arc::new(
        ZoneMapIndex::try_from_serialized(
            batch,
            store.clone(),
            None,
            &cache,
            10,
            None, // legacy: null positions unknown
            true,
            true,
            ZoneMapIndexVersion::V1,
        )
        .unwrap(),
    );
    assert!(legacy_index.null_rows.is_none());

    // Build a seed for fragment 1 with known null positions.
    let rows_per_zone = 10u64;
    let seed_values: ArrayRef = Arc::new(Int32Array::from(vec![
        None, // frag 1, row 0
        Some(1),
    ]));
    let mut seed_writer = ZoneMapSeedWriter::new("value", rows_per_zone, DataType::Int32).unwrap();
    seed_writer.observe_batch(&seed_values).unwrap();
    let seed_bytes = seed_writer
        .finish()
        .unwrap()
        .expect("seed must produce bytes");

    let dest_tmpdir = TempObjDir::default();
    let dest_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        dest_tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    legacy_index
        .try_update_with_seeds(
            &[FragmentSeed {
                fragment_id: 1,
                physical_rows: 2,
                bytes: seed_bytes,
                metadata_value: format!("0:{rows_per_zone}"),
            }],
            dest_store.as_ref(),
        )
        .await
        .unwrap()
        .expect("update must produce a result");

    let updated = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        true,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    assert!(
        updated.null_rows.is_none(),
        "updating a legacy index must not fabricate a null bitmap: \
         the old fragment's null positions were never tracked"
    );

    let result = updated
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        !result.is_exact(),
        "IS NULL on an index derived from a legacy source must use zone scan, not exact lookup"
    );
}

/// Merging a modern index (null_rows = Some) with a legacy index (null_rows = None)
/// must produce a merged index with null_rows = None.  Returning the modern
/// fragment's bitmap as-is would be a false negative for the legacy fragment.
#[tokio::test]
async fn test_merge_modern_and_legacy_index_drops_null_bitmap() {
    use arrow_array::{Int32Array, UInt32Array};

    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let cache = LanceCache::no_cache();

    // Modern index: fragment 0, null_rows = Some(bitmap with 1 known null).
    let schema = Arc::new(Schema::new(vec![
        Field::new("min", DataType::Int32, true),
        Field::new("max", DataType::Int32, true),
        Field::new("null_count", DataType::UInt32, false),
        Field::new("nan_count", DataType::UInt32, false),
        Field::new("negative_nan_count", DataType::UInt32, false),
        Field::new("fragment_id", DataType::UInt64, false),
        Field::new("zone_start", DataType::UInt64, false),
        Field::new("zone_length", DataType::UInt64, false),
    ]));
    let modern_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![Some(1i32)])) as _,
            Arc::new(Int32Array::from(vec![Some(5i32)])) as _,
            Arc::new(UInt32Array::from(vec![1u32])) as _,
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![10u64])) as _,
        ],
    )
    .unwrap();
    let mut modern_null_rows = RowAddrTreeMap::new();
    modern_null_rows.insert(2u64); // frag 0, row 2
    let modern_index = Arc::new(
        ZoneMapIndex::try_from_serialized(
            modern_batch,
            store.clone(),
            None,
            &cache,
            10,
            Some(modern_null_rows),
            false,
            true,
            ZoneMapIndexVersion::V1,
        )
        .unwrap(),
    );

    // Legacy index: fragment 1, null_rows = None.
    let legacy_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![Some(10i32)])) as _,
            Arc::new(Int32Array::from(vec![Some(20i32)])) as _,
            Arc::new(UInt32Array::from(vec![3u32])) as _, // 3 nulls, positions unknown
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt32Array::from(vec![0u32])) as _,
            Arc::new(UInt64Array::from(vec![1u64])) as _,
            Arc::new(UInt64Array::from(vec![0u64])) as _,
            Arc::new(UInt64Array::from(vec![10u64])) as _,
        ],
    )
    .unwrap();
    let legacy_index = Arc::new(
        ZoneMapIndex::try_from_serialized(
            legacy_batch,
            store.clone(),
            None,
            &cache,
            10,
            None, // legacy: null positions unknown
            false,
            true,
            ZoneMapIndexVersion::V1,
        )
        .unwrap(),
    );

    let dest_tmpdir = TempObjDir::default();
    let dest_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        dest_tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let all_frags = RoaringBitmap::from_iter([0u32, 1]);
    merge_zonemap_indices(
        &[modern_index.as_ref(), legacy_index.as_ref()],
        dest_store.as_ref(),
        &all_frags,
    )
    .await
    .unwrap();

    let merged = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    assert!(
        merged.null_rows.is_none(),
        "merging a modern index with a legacy index must drop the null bitmap: \
         the legacy fragment's null positions are unknown"
    );

    let result = merged
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        !result.is_exact(),
        "IS NULL on a merged index with a legacy source must use zone scan, not exact lookup"
    );
}

// ───────────────────────────── Nested type zone map tests ─────────────────────────────

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array};

// --- NaN pruning under Arrow Rust total ordering ----------------------
//
// Total ordering ranks NaNs by sign and payload:
//   -NaN(big payload) < -NaN(canonical) < -inf < finite < +inf
//                     < +NaN(canonical) < +NaN(big payload).
// Zone counts know only how many NaNs of each sign exist, never their
// payloads, so a zone may be pruned only when no payload could match.

const F32_NAN_CANONICAL_BITS: u32 = 0x7FC0_0000;
const F32_NAN_BIG_BITS: u32 = 0x7FFF_FFFF;
const F32_NAN_SIGNALING_BITS: u32 = 0x7F80_0001;
const F32_SIGN_BIT: u32 = 0x8000_0000;

fn f32_pos_nan(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn f32_neg_nan(bits: u32) -> f32 {
    f32::from_bits(bits | F32_SIGN_BIT)
}

/// Zone statistics as a version 1 seed reports them: finite extrema that
/// exclude NaNs, exact sign-split counts.
fn f32_zone_v1(values: &[Option<f32>]) -> ZoneMapStatistics {
    let finite: Vec<f32> = values
        .iter()
        .flatten()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    let nans: Vec<f32> = values
        .iter()
        .flatten()
        .copied()
        .filter(|value| value.is_nan())
        .collect();
    ZoneMapStatistics {
        min: ScalarValue::Float32(finite.iter().copied().reduce(f32::min)),
        max: ScalarValue::Float32(finite.iter().copied().reduce(f32::max)),
        null_count: values.iter().filter(|value| value.is_none()).count() as u32,
        nan_count: nans.len() as u32,
        negative_nan_count: Some(
            nans.iter().filter(|value| value.is_sign_negative()).count() as u32
        ),
        bound: ZoneBound {
            fragment_id: 0,
            start: 0,
            length: values.len(),
        },
    }
}

/// The same zone as a legacy source reports it: NaN presence folded into
/// the NaN max marker, sign split unknown.
fn f32_zone_legacy(values: &[Option<f32>]) -> ZoneMapStatistics {
    let mut zone = f32_zone_v1(values);
    zone.negative_nan_count = None;
    if zone.nan_count > 0 {
        zone.max = ScalarValue::Float32(Some(f32::NAN));
    }
    zone
}

fn f32_bound_value(bound: &Bound<ScalarValue>) -> Option<f32> {
    match bound {
        Bound::Included(ScalarValue::Float32(Some(value)))
        | Bound::Excluded(ScalarValue::Float32(Some(value))) => Some(*value),
        _ => None,
    }
}

/// Whether `value` satisfies the range under total ordering.
fn f32_total_order_in_range(
    value: f32,
    start: &Bound<ScalarValue>,
    end: &Bound<ScalarValue>,
) -> bool {
    use std::cmp::Ordering;
    let lower = match start {
        Bound::Unbounded => true,
        Bound::Included(_) => value.total_cmp(&f32_bound_value(start).unwrap()) != Ordering::Less,
        Bound::Excluded(_) => {
            value.total_cmp(&f32_bound_value(start).unwrap()) == Ordering::Greater
        }
    };
    let upper = match end {
        Bound::Unbounded => true,
        Bound::Included(_) => value.total_cmp(&f32_bound_value(end).unwrap()) != Ordering::Greater,
        Bound::Excluded(_) => value.total_cmp(&f32_bound_value(end).unwrap()) == Ordering::Less,
    };
    lower && upper
}

/// Exhaustive no-false-negative check: a pruned zone must have no row
/// matching the range under total ordering, across NaN payloads and signs,
/// strict and non-strict bounds, and known and unknown sign splits.
#[tokio::test]
async fn test_range_nan_pruning_never_drops_a_matching_zone() {
    let index = train_and_load::<Float32Type>(vec![vec![Some(1.0)]]).await;

    let pos_canonical = f32_pos_nan(F32_NAN_CANONICAL_BITS);
    let pos_big = f32_pos_nan(F32_NAN_BIG_BITS);
    let pos_signaling = f32_pos_nan(F32_NAN_SIGNALING_BITS);
    let neg_canonical = f32_neg_nan(F32_NAN_CANONICAL_BITS);
    let neg_big = f32_neg_nan(F32_NAN_BIG_BITS);

    let value_sets: Vec<Vec<Option<f32>>> = vec![
        vec![Some(pos_canonical); 3],
        vec![Some(neg_canonical); 3],
        vec![Some(pos_canonical), Some(pos_big), Some(pos_signaling)],
        vec![Some(neg_canonical), Some(neg_big)],
        vec![Some(pos_big), Some(neg_big)],
        vec![Some(neg_canonical), None, None],
        vec![Some(pos_canonical), None],
        vec![Some(1.0), Some(2.0), Some(pos_canonical)],
        vec![Some(-3.0), Some(5.0), Some(neg_big)],
        vec![Some(1.0), Some(2.0)],
        vec![None, None],
        vec![Some(neg_big), Some(0.5), Some(pos_big), None],
    ];
    let bound_values = [
        neg_big,
        neg_canonical,
        -2.0_f32,
        0.0_f32,
        2.0_f32,
        pos_canonical,
        pos_big,
    ];
    let mut bounds: Vec<Bound<ScalarValue>> = vec![Bound::Unbounded];
    for value in bound_values {
        bounds.push(Bound::Included(ScalarValue::Float32(Some(value))));
        bounds.push(Bound::Excluded(ScalarValue::Float32(Some(value))));
    }

    for values in &value_sets {
        for zone in [f32_zone_v1(values), f32_zone_legacy(values)] {
            for start in &bounds {
                for end in &bounds {
                    let query = SargableQuery::Range(start.clone(), end.clone());
                    let kept = index.evaluate_zone_against_query(&zone, &query).unwrap();
                    let matchable = values
                        .iter()
                        .flatten()
                        .any(|value| f32_total_order_in_range(*value, start, end));
                    assert!(
                        kept || !matchable,
                        "zone {zone:?} (values {values:?}) wrongly pruned for {query:?}"
                    );
                }
            }
        }
    }
}

/// Targeted cases: zones kept when a NaN payload could match, pruned
/// exactly when the statistics prove no value can.
#[tokio::test]
async fn test_range_nan_pruning_cases() {
    let index = train_and_load::<Float32Type>(vec![vec![Some(1.0)]]).await;
    let eval = |zone: &ZoneMapStatistics, start: Bound<ScalarValue>, end: Bound<ScalarValue>| {
        index
            .evaluate_zone_against_query(zone, &SargableQuery::Range(start, end))
            .unwrap()
    };
    let f32s = |value: f32| ScalarValue::Float32(Some(value));
    let pos_canonical = f32_pos_nan(F32_NAN_CANONICAL_BITS);
    let pos_big = f32_pos_nan(F32_NAN_BIG_BITS);
    let neg_canonical = f32_neg_nan(F32_NAN_CANONICAL_BITS);
    let neg_big = f32_neg_nan(F32_NAN_BIG_BITS);

    // All-negative-NaN zone: matches x < 0 (negative NaNs sort below every
    // finite value), with and without a known sign split.
    let all_neg = [Some(neg_canonical), Some(neg_canonical)];
    for zone in [f32_zone_v1(&all_neg), f32_zone_legacy(&all_neg)] {
        assert!(eval(&zone, Bound::Unbounded, Bound::Excluded(f32s(0.0))));
        // But not x > 0: no negative NaN sorts above a finite bound.
        if zone.negative_nan_count.is_some() {
            assert!(!eval(&zone, Bound::Excluded(f32s(0.0)), Bound::Unbounded));
        } else {
            // Unknown split must stay conservative.
            assert!(eval(&zone, Bound::Excluded(f32s(0.0)), Bound::Unbounded));
        }
    }

    // All-positive-NaN zone: matches x > 0, not x < 0 when the split is known.
    let all_pos = [Some(pos_canonical), Some(pos_big)];
    for zone in [f32_zone_v1(&all_pos), f32_zone_legacy(&all_pos)] {
        assert!(eval(&zone, Bound::Excluded(f32s(0.0)), Bound::Unbounded));
        if zone.negative_nan_count.is_some() {
            assert!(!eval(&zone, Bound::Unbounded, Bound::Excluded(f32s(0.0))));
        }
    }

    // NaN mixed with nulls and no ordinary values: the null rows never
    // match a range, the NaNs follow their sign.
    let nan_and_null = [Some(neg_canonical), None, None];
    let zone = f32_zone_v1(&nan_and_null);
    assert!(eval(&zone, Bound::Unbounded, Bound::Included(f32s(10.0))));
    assert!(!eval(&zone, Bound::Included(f32s(0.0)), Bound::Unbounded));

    // Strict NaN bounds cannot exclude same-sign NaNs: payloads above or
    // below the bound NaN exist.
    let only_pos_big = [Some(pos_big)];
    let zone = f32_zone_v1(&only_pos_big);
    assert!(
        eval(
            &zone,
            Bound::Excluded(f32s(pos_canonical)),
            Bound::Unbounded
        ),
        "x > +NaN(canonical) must keep a zone holding +NaN(bigger payload)"
    );
    let only_neg_big = [Some(neg_big)];
    let zone = f32_zone_v1(&only_neg_big);
    assert!(
        eval(
            &zone,
            Bound::Unbounded,
            Bound::Excluded(f32s(neg_canonical))
        ),
        "x < -NaN(canonical) must keep a zone holding -NaN(bigger payload)"
    );
    // While the opposite sign is excluded exactly.
    let zone = f32_zone_v1(&all_neg);
    assert!(!eval(
        &zone,
        Bound::Excluded(f32s(pos_canonical)),
        Bound::Unbounded
    ));

    // A two-sided finite range: neither NaN sign can satisfy both sides,
    // so known statistics prune exactly on the finite extrema.
    let finite_and_nan = [Some(1.0), Some(2.0), Some(pos_canonical)];
    let zone = f32_zone_v1(&finite_and_nan);
    assert!(eval(
        &zone,
        Bound::Included(f32s(0.0)),
        Bound::Included(f32s(5.0))
    ));
    assert!(
        !eval(
            &zone,
            Bound::Included(f32s(10.0)),
            Bound::Included(f32s(20.0))
        ),
        "a two-sided finite range beyond the finite extrema must prune, NaNs included"
    );
    // The legacy marker hides the finite max, so only the min side prunes.
    let zone = f32_zone_legacy(&finite_and_nan);
    assert!(eval(
        &zone,
        Bound::Included(f32s(10.0)),
        Bound::Included(f32s(20.0))
    ));
    assert!(!eval(
        &zone,
        Bound::Included(f32s(-10.0)),
        Bound::Included(f32s(0.5))
    ));
}

/// The other float widths share the sign logic: canonical and payload
/// NaNs keep and prune the same way for Float16 and Float64.
#[tokio::test]
async fn test_range_nan_pruning_other_float_widths() {
    // Float64: a zone of negative NaNs matches x < 0, not x > 0.
    let index = train_and_load::<Float64Type>(vec![vec![Some(1.0)]]).await;
    let zone = ZoneMapStatistics {
        min: ScalarValue::Float64(None),
        max: ScalarValue::Float64(None),
        null_count: 0,
        nan_count: 2,
        negative_nan_count: Some(2),
        bound: ZoneBound {
            fragment_id: 0,
            start: 0,
            length: 2,
        },
    };
    let query = SargableQuery::Range(
        Bound::Unbounded,
        Bound::Excluded(ScalarValue::Float64(Some(0.0))),
    );
    assert!(index.evaluate_zone_against_query(&zone, &query).unwrap());
    let query = SargableQuery::Range(
        Bound::Excluded(ScalarValue::Float64(Some(0.0))),
        Bound::Unbounded,
    );
    assert!(!index.evaluate_zone_against_query(&zone, &query).unwrap());
    // Strict bound against a same-sign payload NaN keeps the zone.
    let query = SargableQuery::Range(
        Bound::Unbounded,
        Bound::Excluded(ScalarValue::Float64(Some(f64::from_bits(
            0xFFF8_0000_0000_0000,
        )))),
    );
    assert!(index.evaluate_zone_against_query(&zone, &query).unwrap());

    // Float16: a positive-NaN zone keeps x > +NaN (payload order) and
    // prunes x < 0 when the split is known.
    let index16 = train_and_load::<arrow_array::types::Float16Type>(vec![vec![Some(
        half::f16::from_f32(1.0),
    )]])
    .await;
    let zone = ZoneMapStatistics {
        min: ScalarValue::Float16(None),
        max: ScalarValue::Float16(None),
        null_count: 0,
        nan_count: 1,
        negative_nan_count: Some(0),
        bound: ZoneBound {
            fragment_id: 0,
            start: 0,
            length: 1,
        },
    };
    let query = SargableQuery::Range(
        Bound::Excluded(ScalarValue::Float16(Some(half::f16::NAN))),
        Bound::Unbounded,
    );
    assert!(index16.evaluate_zone_against_query(&zone, &query).unwrap());
    let query = SargableQuery::Range(
        Bound::Unbounded,
        Bound::Excluded(ScalarValue::Float16(Some(half::f16::from_f32(0.0)))),
    );
    assert!(!index16.evaluate_zone_against_query(&zone, &query).unwrap());
}

/// Equals and IsIn with NaN targets follow the sign counts, payload
/// conservative.
#[tokio::test]
async fn test_equals_nan_payload_targets() {
    let index = train_and_load::<Float32Type>(vec![vec![Some(1.0)]]).await;
    let pos_big = f32_pos_nan(F32_NAN_BIG_BITS);
    let neg_canonical = f32_neg_nan(F32_NAN_CANONICAL_BITS);

    let only_neg = [Some(neg_canonical)];
    let zone = f32_zone_v1(&only_neg);
    let query = SargableQuery::Equals(ScalarValue::Float32(Some(pos_big)));
    assert!(
        !index.evaluate_zone_against_query(&zone, &query).unwrap(),
        "a +NaN target cannot match a zone with only negative NaNs"
    );
    let query = SargableQuery::Equals(ScalarValue::Float32(Some(f32_neg_nan(F32_NAN_BIG_BITS))));
    assert!(
        index.evaluate_zone_against_query(&zone, &query).unwrap(),
        "a -NaN target keeps a zone with negative NaNs regardless of payload"
    );
    let legacy = f32_zone_legacy(&only_neg);
    let query = SargableQuery::IsIn(vec![ScalarValue::Float32(Some(pos_big))]);
    assert!(
        index.evaluate_zone_against_query(&legacy, &query).unwrap(),
        "an unknown sign split stays conservative for NaN targets"
    );
}

// --- Binary-copy seed parsing and merging ------------------------------

fn seed_bytes_f32(values: &[Option<f32>], rows_per_zone: u64) -> bytes::Bytes {
    use crate::scalar::seed::IndexSeedWriter;
    let mut writer = ZoneMapSeedWriter::new("val", rows_per_zone, DataType::Float32).unwrap();
    let array: ArrayRef = Arc::new(arrow_array::Float32Array::from(values.to_vec()));
    writer.observe_batch(&array).unwrap();
    writer.finish().unwrap().unwrap()
}

/// Rewrite a seed buffer's schema metadata and zone batch, to fabricate
/// version 0 buffers and corrupt inputs.
fn transform_seed(
    bytes: &bytes::Bytes,
    transform: impl FnOnce(&mut std::collections::HashMap<String, String>, RecordBatch) -> RecordBatch,
) -> bytes::Bytes {
    use arrow_ipc::reader::FileReader as IpcReader;
    use arrow_ipc::writer::FileWriter as IpcWriter;
    use std::io::Cursor;

    let mut reader = IpcReader::try_new(Cursor::new(bytes.as_ref()), None).unwrap();
    let mut metadata = reader.schema().metadata.clone();
    let batch = reader.next().unwrap().unwrap();
    let batch = transform(&mut metadata, batch);
    let mut schema = batch.schema().as_ref().clone();
    schema.metadata = metadata;
    let batch = RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap();
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = IpcWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    bytes::Bytes::from(buf.into_inner())
}

fn to_v0(bytes: &bytes::Bytes) -> bytes::Bytes {
    transform_seed(bytes, |metadata, batch| {
        metadata.remove(SEED_VERSION_META_KEY);
        metadata.remove(SEED_NULL_BITMAP_META_KEY);
        let keep: Vec<usize> = batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() != "negative_nan_count")
            .map(|(i, _)| i)
            .collect();
        batch.project(&keep).unwrap()
    })
}

fn set_zone_lengths(bytes: &bytes::Bytes, lengths: &[u64]) -> bytes::Bytes {
    transform_seed(bytes, |_, batch| {
        let columns: Vec<ArrayRef> = batch
            .schema()
            .fields()
            .iter()
            .zip(batch.columns())
            .map(|(field, col)| {
                if field.name() == "zone_length" {
                    Arc::new(UInt64Array::from(lengths.to_vec())) as ArrayRef
                } else {
                    col.clone()
                }
            })
            .collect();
        RecordBatch::try_new(batch.schema(), columns).unwrap()
    })
}

/// Merged ragged sources: zone starts are the cumulative lengths, with
/// short zones mid file, and the statistics equal a concatenation.
#[test]
fn test_merge_seeds_cumulative_starts() {
    let a_values: Vec<Option<f32>> = vec![
        Some(1.0),
        None,
        Some(-f32::NAN),
        Some(3.0),
        // Tail zone of 2 rows under rows_per_zone = 4.
        Some(f32::NAN),
        None,
    ];
    let b_values: Vec<Option<f32>> = vec![Some(10.0), Some(-2.5), None];
    // Different collection granularities concatenate: zones are located
    // by their lengths, and the caller picks the output locator's value.
    let a = seed_bytes_f32(&a_values, 4);
    let b = seed_bytes_f32(&b_values, 8);
    let parsed = [parse_supported(&a, 4), parse_supported(&b, 8)];
    assert_eq!(
        (parsed[0].rows_per_zone(), parsed[1].rows_per_zone()),
        (4, 8)
    );
    assert_eq!(parsed[0].total_rows(), 6);
    assert_eq!(parsed[1].total_rows(), 3);

    let merged = merge_seeds_for_binary_copy(&[&parsed[0], &parsed[1]])
        .unwrap()
        .unwrap();
    // Zones are located by length; the locator's rows_per_zone is never
    // a read-time stride (a source zone may even exceed it).
    let (zones, bitmap) = ZoneMapSeedWriter::deserialize_seed(7, &merged).unwrap();
    let bounds: Vec<(u64, usize)> = zones
        .iter()
        .map(|z| (z.bound.start, z.bound.length))
        .collect();
    assert_eq!(bounds, vec![(0, 4), (4, 2), (6, 3)]);
    assert!(zones.iter().all(|z| z.bound.fragment_id == 7));
    assert_eq!(
        zones.iter().map(|z| z.null_count).collect::<Vec<_>>(),
        vec![1, 1, 1]
    );
    assert_eq!(
        zones.iter().map(|z| z.nan_count).collect::<Vec<_>>(),
        vec![1, 1, 0]
    );
    assert_eq!(
        zones
            .iter()
            .map(|z| z.negative_nan_count)
            .collect::<Vec<_>>(),
        vec![Some(1), Some(0), Some(0)]
    );
    // Null offsets shifted by each source's cumulative rows: 1, 5, 6 + 2.
    let bitmap = bitmap.unwrap();
    assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![1, 5, 8]);
}

/// Any version 0 source refuses the merge; a bitmap missing on one source
/// leaves the output without one.
fn parse_supported(bytes: &bytes::Bytes, rows_per_zone: u64) -> MergeableSeed {
    match parse_seed_for_binary_copy(bytes, rows_per_zone).unwrap() {
        CarriedSeed::Supported(seed) => seed,
        CarriedSeed::UnsupportedVersion(version) => {
            panic!("expected a supported seed, got version {version}")
        }
    }
}

#[test]
fn test_merge_seeds_version_and_bitmap_rules() {
    let a = seed_bytes_f32(&[Some(1.0), Some(2.0)], 2);
    let b = seed_bytes_f32(&[Some(3.0), None], 2);
    let a_parsed = parse_supported(&a, 2);

    let b_v0 = to_v0(&b);
    let b_v0_parsed = parse_supported(&b_v0, 2);
    assert!(!b_v0_parsed.is_version_1());
    assert!(
        merge_seeds_for_binary_copy(&[&a_parsed, &b_v0_parsed])
            .unwrap()
            .is_none(),
        "version 0 sources are never converted"
    );

    // A version 1 source without a bitmap (older writer) merges, but the
    // output then has no bitmap: absence is unknown, not zero.
    let b_no_bitmap = transform_seed(&b, |metadata, batch| {
        metadata.remove(SEED_NULL_BITMAP_META_KEY);
        batch
    });
    let b_parsed = parse_supported(&b_no_bitmap, 2);
    let merged = merge_seeds_for_binary_copy(&[&a_parsed, &b_parsed])
        .unwrap()
        .unwrap();
    let (zones, bitmap) = ZoneMapSeedWriter::deserialize_seed(0, &merged).unwrap();
    assert!(bitmap.is_none());
    assert_eq!(
        zones.iter().map(|z| z.null_count).collect::<Vec<_>>(),
        vec![0, 1],
        "exact null counts survive a missing bitmap"
    );
}

/// Corrupt or contradictory buffers refuse parsing instead of posing as
/// missing statistics; a future version is recognized but never
/// interpreted, so no merge can relabel it as version 1.
#[test]
fn test_parse_seed_rejects_corrupt_and_unknown() {
    let good = seed_bytes_f32(&[Some(1.0), Some(2.0), Some(3.0)], 2);
    let good_parsed = parse_supported(&good, 2);

    // A version 2 fixture: version 1 bytes relabeled, which a version 1
    // build must not read as version 1 even though the columns match.
    let future_version = transform_seed(&good, |metadata, batch| {
        metadata.insert(SEED_VERSION_META_KEY.to_string(), "2".to_string());
        batch
    });
    let CarriedSeed::UnsupportedVersion(version) =
        parse_seed_for_binary_copy(&future_version, 2).unwrap()
    else {
        panic!("a future version must not parse as supported statistics");
    };
    assert_eq!(version, "2");
    // Only supported seeds can reach the merge; a version 1 source alone
    // merges to itself, and the decoder refuses the version 2 bytes.
    assert!(
        merge_seeds_for_binary_copy(&[&good_parsed])
            .unwrap()
            .is_some()
    );
    let error = ZoneMapSeedWriter::deserialize_seed(0, &future_version).unwrap_err();
    assert!(matches!(error, lance_core::Error::NotSupported { .. }));

    // A future version may also change the bitmap encoding: the version
    // is refused before any other metadata is interpreted, so the error
    // names the version, not a corrupt bitmap.
    let future_bitmap = transform_seed(&good, |metadata, batch| {
        metadata.insert(SEED_VERSION_META_KEY.to_string(), "2".to_string());
        metadata.insert(
            SEED_NULL_BITMAP_META_KEY.to_string(),
            "not-hex-in-version-2".to_string(),
        );
        batch
    });
    let error = ZoneMapSeedWriter::deserialize_seed(0, &future_bitmap).unwrap_err();
    assert!(
        matches!(error, lance_core::Error::NotSupported { .. }),
        "{error}"
    );
    assert!(matches!(
        parse_seed_for_binary_copy(&future_bitmap, 2).unwrap(),
        CarriedSeed::UnsupportedVersion(_)
    ));

    let zero_length = set_zone_lengths(&good, &[2, 0]);
    assert!(parse_seed_for_binary_copy(&zero_length, 2).is_err());

    // Counts exceeding the zone length contradict the statistics.
    let shrunk = set_zone_lengths(&good, &[2, 1]);
    let lying = transform_seed(&shrunk, |_, batch| {
        let columns: Vec<ArrayRef> = batch
            .schema()
            .fields()
            .iter()
            .zip(batch.columns())
            .map(|(field, col)| {
                if field.name() == "null_count" {
                    Arc::new(UInt32Array::from(vec![0u32, 5])) as ArrayRef
                } else {
                    col.clone()
                }
            })
            .collect();
        RecordBatch::try_new(batch.schema(), columns).unwrap()
    });
    assert!(parse_seed_for_binary_copy(&lying, 2).is_err());

    assert!(
        parse_seed_for_binary_copy(&bytes::Bytes::from_static(b"not an ipc buffer"), 2).is_err()
    );
    assert!(
        parse_seed_for_binary_copy(&good, 0).is_err(),
        "a locator must declare a non-zero rows_per_zone"
    );

    // A version 0 buffer parses (so the gate can account for it) but is
    // never carried.
    let v0 = to_v0(&good);
    let v0_parsed = parse_supported(&v0, 2);
    assert!(!v0_parsed.is_version_1());
}

/// deserialize_seed locates every zone by the cumulative lengths, for
/// both versions: a well-formed fixed-stride seed reads unchanged, and
/// mid-file short zones read at their true offsets.
#[test]
fn test_deserialize_seed_locates_zones_by_length() {
    let values: Vec<Option<f32>> = (0..6).map(|i| Some(i as f32)).collect();
    let v1 = seed_bytes_f32(&values, 4);
    let (zones, _) = ZoneMapSeedWriter::deserialize_seed(0, &v1).unwrap();
    assert_eq!(
        zones
            .iter()
            .map(|z| (z.bound.start, z.bound.length))
            .collect::<Vec<_>>(),
        vec![(0, 4), (4, 2)]
    );
    let v1_ragged = set_zone_lengths(&v1, &[2, 4]);
    let (zones, _) = ZoneMapSeedWriter::deserialize_seed(0, &v1_ragged).unwrap();
    assert_eq!(
        zones
            .iter()
            .map(|z| (z.bound.start, z.bound.length))
            .collect::<Vec<_>>(),
        vec![(0, 2), (2, 4)]
    );
    // A zone longer than the declared rows_per_zone (a source collected at
    // a coarser granularity) is still located by its length.
    let (zones, _) = ZoneMapSeedWriter::deserialize_seed(0, &v1).unwrap();
    assert_eq!(
        zones
            .iter()
            .map(|z| (z.bound.start, z.bound.length))
            .collect::<Vec<_>>(),
        vec![(0, 4), (4, 2)]
    );

    let v0 = to_v0(&v1);
    let (zones, _) = ZoneMapSeedWriter::deserialize_seed(0, &v0).unwrap();
    assert_eq!(
        zones
            .iter()
            .map(|z| (z.bound.start, z.bound.length))
            .collect::<Vec<_>>(),
        vec![(0, 4), (4, 2)]
    );
}

/// Build a FixedSizeList<Float32, list_size> zone map from `rows` (each row is a
/// `Vec<Option<f32>>` of length `list_size`), then load and return the index.
async fn train_and_load_fsl(rows: Vec<Vec<Option<f32>>>, list_size: i32) -> Arc<ZoneMapIndex> {
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let fsl_type = DataType::FixedSizeList(item_field.clone(), list_size);
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, fsl_type.clone(), true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));

    let mut fsl_builder =
        arrow_array::builder::FixedSizeListBuilder::new(Float32Builder::new(), list_size);
    for row in &rows {
        assert_eq!(row.len(), list_size as usize);
        for &v in row {
            match v {
                Some(f) => fsl_builder.values().append_value(f),
                None => fsl_builder.values().append_null(),
            }
        }
        fsl_builder.append(true);
    }
    let fsl_arr: ArrayRef = Arc::new(fsl_builder.finish());
    let n = rows.len() as u64;
    let row_addr: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..n));
    let batch = RecordBatch::try_new(schema.clone(), vec![fsl_arr, row_addr]).unwrap();

    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::once(std::future::ready(Ok(batch))),
    ));

    ZoneMapIndexPlugin::train_zonemap_index(
        stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(4)),
    )
    .await
    .unwrap();

    ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .expect("failed to load ZoneMapIndex")
}

use arrow_array::builder::Float32Builder;
use datafusion_common::ScalarValue as SV;

fn fsl_scalar(values: Vec<Option<f32>>) -> SV {
    let list_size = values.len() as i32;
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let items: ArrayRef = Arc::new(Float32Array::from(values));
    let arr = Arc::new(FixedSizeListArray::new(item_field, list_size, items, None));
    SV::FixedSizeList(arr)
}

// Zones for nested types have null min/max; all non-null queries are conservative (not pruned)
// unless every row in the zone is null.
#[tokio::test]
async fn test_fsl_zonemap_conservative_equals() {
    // Two zones, 4 rows each, no null rows.
    let index = train_and_load_fsl(
        vec![
            vec![Some(1.0), Some(2.0)],
            vec![Some(3.0), Some(4.0)],
            vec![Some(5.0), Some(6.0)],
            vec![Some(7.0), Some(8.0)],
            vec![Some(10.0), Some(20.0)],
            vec![Some(30.0), Some(40.0)],
            vec![Some(50.0), Some(60.0)],
            vec![Some(70.0), Some(80.0)],
        ],
        2,
    )
    .await;
    assert_eq!(index.zones.len(), 2);
    // min/max are null for nested types
    assert!(index.zones[0].min.is_null());
    assert!(index.zones[0].max.is_null());
    assert_eq!(index.zones[0].null_count, 0);
    assert_eq!(index.zones[0].nan_count, 0);

    // Non-null Equals: conservative — neither zone is pruned
    let q = SargableQuery::Equals(fsl_scalar(vec![Some(100.0), Some(200.0)]));
    assert!(
        index
            .evaluate_zone_against_query(&index.zones[0], &q)
            .unwrap(),
        "non-null Equals on nested type is conservative"
    );
    assert!(
        index
            .evaluate_zone_against_query(&index.zones[1], &q)
            .unwrap(),
        "non-null Equals on nested type is conservative"
    );

    // IsNull: no null rows → both zones pruned
    let q_null = SargableQuery::IsNull();
    assert!(
        !index
            .evaluate_zone_against_query(&index.zones[0], &q_null)
            .unwrap(),
        "IsNull pruned when null_count=0"
    );
    assert!(
        !index
            .evaluate_zone_against_query(&index.zones[1], &q_null)
            .unwrap(),
        "IsNull pruned when null_count=0"
    );
}

#[tokio::test]
async fn test_fsl_zonemap_null_list() {
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let fsl_type = DataType::FixedSizeList(item_field.clone(), 2);
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, fsl_type.clone(), true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));

    // Build FSL array with some null entries
    let mut builder = arrow_array::builder::FixedSizeListBuilder::new(Float32Builder::new(), 2);
    // Row 0: null list
    builder.values().append_value(0.0);
    builder.values().append_value(0.0);
    builder.append(false);
    // Row 1: [3.0, 4.0]
    builder.values().append_value(3.0);
    builder.values().append_value(4.0);
    builder.append(true);
    let fsl_arr: ArrayRef = Arc::new(builder.finish());
    let row_addr: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..2));
    let batch = RecordBatch::try_new(schema.clone(), vec![fsl_arr, row_addr]).unwrap();

    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::once(std::future::ready(Ok(batch))),
    ));
    ZoneMapIndexPlugin::train_zonemap_index(
        stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(4)),
    )
    .await
    .unwrap();
    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    assert_eq!(index.zones.len(), 1);
    assert_eq!(index.zones[0].null_count, 1, "one null list");

    // IsNull should match
    let q = SargableQuery::IsNull();
    assert!(
        index
            .evaluate_zone_against_query(&index.zones[0], &q)
            .unwrap()
    );

    // Equals null should match (null_count > 0)
    let null_scalar = SV::FixedSizeList(Arc::new(FixedSizeListArray::new_null(
        item_field.clone(),
        2,
        1,
    )));
    let q2 = SargableQuery::Equals(null_scalar);
    assert!(
        index
            .evaluate_zone_against_query(&index.zones[0], &q2)
            .unwrap()
    );
}

#[tokio::test]
async fn test_fsl_zonemap_is_in_conservative() {
    // Two zones (4 rows each), no null rows.
    let index = train_and_load_fsl(
        vec![
            vec![Some(1.0), Some(2.0)],
            vec![Some(3.0), Some(4.0)],
            vec![Some(5.0), Some(6.0)],
            vec![Some(7.0), Some(8.0)],
            vec![Some(10.0), Some(20.0)],
            vec![Some(30.0), Some(40.0)],
            vec![Some(50.0), Some(60.0)],
            vec![Some(70.0), Some(80.0)],
        ],
        2,
    )
    .await;
    assert_eq!(index.zones.len(), 2);

    // IsIn with non-null values: conservative — neither zone is pruned
    let q = SargableQuery::IsIn(vec![
        fsl_scalar(vec![Some(4.0), Some(5.0)]),
        fsl_scalar(vec![Some(100.0), Some(200.0)]),
    ]);
    assert!(
        index
            .evaluate_zone_against_query(&index.zones[0], &q)
            .unwrap(),
        "IsIn with non-null values is conservative for nested types"
    );
    assert!(
        index
            .evaluate_zone_against_query(&index.zones[1], &q)
            .unwrap(),
        "IsIn with non-null values is conservative for nested types"
    );
}

#[tokio::test]
async fn test_fsl_zonemap_value_range_is_none() {
    let index = train_and_load_fsl(
        vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0), Some(4.0)]],
        2,
    )
    .await;
    assert_eq!(index.value_range(), None, "FSL index has no scalar range");
}

#[tokio::test]
async fn test_fsl_zonemap_all_null_zone() {
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let fsl_type = DataType::FixedSizeList(item_field.clone(), 2);
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, fsl_type.clone(), true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let mut builder = arrow_array::builder::FixedSizeListBuilder::new(Float32Builder::new(), 2);
    for _ in 0..2 {
        builder.values().append_value(0.0);
        builder.values().append_value(0.0);
        builder.append(false); // null list
    }
    let fsl_arr: ArrayRef = Arc::new(builder.finish());
    let row_addr: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..2));
    let batch = RecordBatch::try_new(schema.clone(), vec![fsl_arr, row_addr]).unwrap();
    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::once(std::future::ready(Ok(batch))),
    ));
    ZoneMapIndexPlugin::train_zonemap_index(
        stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(4)),
    )
    .await
    .unwrap();
    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    assert_eq!(index.zones[0].null_count, 2);
    assert!(index.zones[0].min.is_null(), "all-null zone: min is null");

    // When every row is null (null_count == zone length), a non-null query is pruned.
    let q = SargableQuery::Equals(fsl_scalar(vec![Some(1.0), Some(2.0)]));
    assert!(
        !index
            .evaluate_zone_against_query(&index.zones[0], &q)
            .unwrap(),
        "all-null zone (null_count == length) is pruned for non-null target"
    );

    // Range and IsIn are also pruned when all rows are null.
    let q_range = SargableQuery::Range(std::ops::Bound::Unbounded, std::ops::Bound::Unbounded);
    assert!(
        !index
            .evaluate_zone_against_query(&index.zones[0], &q_range)
            .unwrap(),
        "all-null zone is pruned for Range query"
    );
}

/// Build a zone map on a Struct column and verify training succeeds and stats are correct.
#[tokio::test]
async fn test_struct_zonemap_null_tracking() {
    use arrow_array::StructArray;
    use arrow_schema::Fields;

    let item_field = Arc::new(Field::new("x", DataType::Int32, true));
    let struct_type = DataType::Struct(Fields::from(vec![item_field.clone()]));
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, struct_type.clone(), true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));

    // 3 non-null struct rows
    let x_values: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![10, 20, 30]));
    let struct_arr: ArrayRef = Arc::new(StructArray::from(vec![(item_field.clone(), x_values)]));
    let row_addr: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..3));
    let batch = RecordBatch::try_new(schema.clone(), vec![struct_arr, row_addr]).unwrap();

    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::once(std::future::ready(Ok(batch))),
    ));
    ZoneMapIndexPlugin::train_zonemap_index(
        stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(10)),
    )
    .await
    .unwrap();

    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    assert_eq!(index.zones.len(), 1);
    assert_eq!(index.zones[0].null_count, 0, "no null struct rows");
    assert_eq!(index.zones[0].nan_count, 0);
    assert!(index.zones[0].min.is_null());
    assert!(index.zones[0].max.is_null());

    // IsNull: no null rows → zone is pruned
    assert!(
        !index
            .evaluate_zone_against_query(&index.zones[0], &SargableQuery::IsNull())
            .unwrap()
    );

    // Non-null Equals: conservative (not pruned) since not all rows are null
    assert!(
        index
            .evaluate_zone_against_query(
                &index.zones[0],
                &SargableQuery::Equals(SV::Int32(Some(99)))
            )
            .unwrap()
    );

    // value_range: always None for nested types
    assert_eq!(index.value_range(), None);
}

/// Build a zone map on a List<Int32> column and verify null tracking works.
#[tokio::test]
async fn test_list_zonemap_null_tracking() {
    use arrow_array::builder::ListBuilder;

    let item_field = Arc::new(Field::new("item", DataType::Int32, true));
    let list_type = DataType::List(item_field.clone());
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, list_type.clone(), true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));

    // Build: [null, [1,2], [3]]
    let mut builder = ListBuilder::new(arrow_array::builder::Int32Builder::new());
    builder.append_null();
    builder.values().append_value(1);
    builder.values().append_value(2);
    builder.append(true);
    builder.values().append_value(3);
    builder.append(true);
    let list_arr: ArrayRef = Arc::new(builder.finish());
    let row_addr: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..3));
    let batch = RecordBatch::try_new(schema.clone(), vec![list_arr, row_addr]).unwrap();

    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::once(std::future::ready(Ok(batch))),
    ));
    ZoneMapIndexPlugin::train_zonemap_index(
        stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(10)),
    )
    .await
    .unwrap();

    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    assert_eq!(index.zones.len(), 1);
    assert_eq!(index.zones[0].null_count, 1, "one null list row");
    // min/max are typed null ScalarValues for nested types

    // IsNull search: the null bitmap is populated during training so
    // the result is an exact set containing just the null row (row address 0).
    let result = index
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    let mut exact_nulls = RowAddrTreeMap::new();
    exact_nulls.insert(0); // only row 0 is null
    assert_eq!(result, SearchResult::exact(exact_nulls));
}
/// Builds a `List<Int32>` with one parent-null row, an empty list, a list
/// containing a null child and a plain list: exactly one null row.
fn nested_list_fixture() -> ArrayRef {
    use arrow_array::builder::{Int32Builder, ListBuilder};
    let mut builder = ListBuilder::new(Int32Builder::new());
    builder.append_null();
    builder.append(true);
    builder.values().append_null();
    builder.values().append_value(1);
    builder.append(true);
    builder.values().append_value(2);
    builder.values().append_value(3);
    builder.append(true);
    Arc::new(builder.finish())
}

fn zone_bound(length: usize) -> ZoneBound {
    ZoneBound {
        fragment_id: 0,
        start: 0,
        length,
    }
}

fn process_one_zone(array: &ArrayRef) -> ZoneMapStatistics {
    let mut processor = ZoneMapProcessor::new(array.data_type().clone()).unwrap();
    processor.process_chunk(array).unwrap();
    processor.finish_zone(zone_bound(array.len())).unwrap()
}

/// Nested zone statistics count parent-level nulls only: empty lists,
/// null children and NaN children leave null_count and nan_count alone.
#[test]
fn test_nested_zone_stats_count_parent_nulls_only() {
    use arrow::buffer::NullBuffer;
    use arrow_array::StructArray;
    use arrow_array::builder::{
        FixedSizeListBuilder, Float32Builder, Int32Builder, LargeListBuilder, MapBuilder,
        StringBuilder,
    };

    let mut cases: Vec<(&str, ArrayRef)> = vec![("list", nested_list_fixture())];

    let mut large = LargeListBuilder::new(Int32Builder::new());
    large.append_null();
    large.append(true);
    large.values().append_null();
    large.values().append_value(1);
    large.append(true);
    large.values().append_value(2);
    large.append(true);
    cases.push(("large_list", Arc::new(large.finish())));

    // [null, [NaN, 1.0], [null, null]]
    let mut fsl = FixedSizeListBuilder::new(Float32Builder::new(), 2);
    fsl.values().append_null();
    fsl.values().append_null();
    fsl.append(false);
    fsl.values().append_value(f32::NAN);
    fsl.values().append_value(1.0);
    fsl.append(true);
    fsl.values().append_null();
    fsl.values().append_null();
    fsl.append(true);
    cases.push(("fixed_size_list", Arc::new(fsl.finish())));

    // Row 0 is a null struct, row 1 is a valid struct whose field is null.
    let x_field = Arc::new(Field::new("x", DataType::Int32, true));
    let x: ArrayRef = Arc::new(Int32Array::from(vec![None, None, Some(5)]));
    let strukt = StructArray::try_new(
        vec![x_field].into(),
        vec![x],
        Some(NullBuffer::from(vec![false, true, true])),
    )
    .unwrap();
    cases.push(("struct", Arc::new(strukt)));

    let mut map = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    map.append(false).unwrap();
    map.keys().append_value("k");
    map.values().append_null();
    map.append(true).unwrap();
    map.append(true).unwrap();
    cases.push(("map", Arc::new(map.finish())));

    for (name, array) in cases {
        assert!(array.data_type().is_nested(), "{name}");
        let stats = process_one_zone(&array);
        assert_eq!(stats.null_count, 1, "{name}: only the parent null counts");
        assert_eq!(stats.nan_count, 0, "{name}: nested NaN is not tracked");
        assert!(stats.min.is_null(), "{name}: nested min is a typed null");
        assert!(stats.max.is_null(), "{name}: nested max is a typed null");
        assert_eq!(stats.min.data_type(), *array.data_type(), "{name}");
        assert_eq!(stats.bound.length, array.len(), "{name}");
    }

    // The count accumulates across chunks and is cleared by reset.
    let list = nested_list_fixture();
    let mut processor = ZoneMapProcessor::new(list.data_type().clone()).unwrap();
    processor.process_chunk(&list).unwrap();
    processor.process_chunk(&list.slice(0, 2)).unwrap();
    assert_eq!(processor.finish_zone(zone_bound(6)).unwrap().null_count, 2);
    processor.reset().unwrap();
    processor.process_chunk(&list.slice(1, 3)).unwrap();
    assert_eq!(processor.finish_zone(zone_bound(3)).unwrap().null_count, 0);
}

/// The null-only path keeps the accumulator's type guard.
#[test]
fn test_nested_zone_stats_reject_type_mismatch() {
    use crate::scalar::seed::IndexSeedWriter;
    use arrow_array::builder::{Int64Builder, ListBuilder};

    let list_i32 = nested_list_fixture();
    let mut wide = ListBuilder::new(Int64Builder::new());
    wide.values().append_value(1);
    wide.append(true);
    let list_i64: ArrayRef = Arc::new(wide.finish());
    let plain: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));

    let mut processor = ZoneMapProcessor::new(list_i32.data_type().clone()).unwrap();
    for wrong in [&list_i64, &plain] {
        let err = processor.process_chunk(wrong).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "unexpected error: {err:?}"
        );
        assert!(err.to_string().contains("Type mismatch"), "{err}");
    }
    // A rejected chunk leaves no trace.
    processor.process_chunk(&list_i32).unwrap();
    assert_eq!(processor.finish_zone(zone_bound(4)).unwrap().null_count, 1);

    let mut writer = ZoneMapSeedWriter::new("c", 4, list_i32.data_type().clone()).unwrap();
    let err = writer.observe_batch(&plain).unwrap_err();
    assert!(err.to_string().contains("Type mismatch"), "{err}");
}

/// Seeds for a nested column carry parent-level null counts per zone and
/// exact parent-null positions, regardless of how the rows are batched.
#[test]
fn test_nested_seed_writer_null_only_round_trip() {
    use crate::scalar::seed::IndexSeedWriter;
    use arrow_array::builder::{Int32Builder, ListBuilder};

    // Row shapes: N = null row, E = empty list, C = list with a null child, V = values.
    let shapes = ['N', 'E', 'C', 'V', 'N', 'N', 'N', 'V', 'E', 'C', 'N'];
    let mut builder = ListBuilder::new(Int32Builder::new());
    for (i, shape) in shapes.iter().enumerate() {
        match shape {
            'N' => builder.append_null(),
            'E' => builder.append(true),
            'C' => {
                builder.values().append_null();
                builder.values().append_value(i as i32);
                builder.append(true);
            }
            _ => {
                builder.values().append_value(i as i32);
                builder.values().append_value(i as i32 + 1);
                builder.append(true);
            }
        }
    }
    let array: ArrayRef = Arc::new(builder.finish());
    let expected_nulls: RoaringBitmap = shapes
        .iter()
        .enumerate()
        .filter(|(_, s)| **s == 'N')
        .map(|(i, _)| i as u32)
        .collect();

    let rows_per_zone = 4;
    let mut whole = ZoneMapSeedWriter::new("c", rows_per_zone, array.data_type().clone()).unwrap();
    whole.observe_batch(&array).unwrap();
    let mut sliced = ZoneMapSeedWriter::new("c", rows_per_zone, array.data_type().clone()).unwrap();
    for (start, len) in [(0, 3), (3, 2), (5, 6)] {
        sliced.observe_batch(&array.slice(start, len)).unwrap();
    }
    let whole_bytes = whole.finish().unwrap().unwrap();
    let sliced_bytes = sliced.finish().unwrap().unwrap();
    assert_eq!(
        whole_bytes, sliced_bytes,
        "batching must not change the seed"
    );

    let (zones, bitmap) = ZoneMapSeedWriter::deserialize_seed(7, &whole_bytes).unwrap();
    assert_eq!(bitmap, Some(expected_nulls));
    let lengths: Vec<usize> = zones.iter().map(|z| z.bound.length).collect();
    assert_eq!(lengths, vec![4, 4, 3]);
    let null_counts: Vec<u32> = zones.iter().map(|z| z.null_count).collect();
    assert_eq!(null_counts, vec![1, 3, 1]);
    for zone in &zones {
        assert_eq!(zone.bound.fragment_id, 7);
        assert_eq!(zone.nan_count, 0);
        assert!(zone.min.is_null() && zone.max.is_null());
        assert_eq!(zone.min.data_type(), *array.data_type());
    }
}

/// Training on a list column ignores empty lists and null children when
/// deciding which rows are null.
#[tokio::test]
async fn test_list_zonemap_training_ignores_child_nulls() {
    let list_arr = nested_list_fixture();
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, list_arr.data_type().clone(), true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let row_addr: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..4));
    let batch = RecordBatch::try_new(schema.clone(), vec![list_arr, row_addr]).unwrap();

    let tmpdir = TempObjDir::default();
    let test_store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::once(std::future::ready(Ok(batch))),
    ));
    ZoneMapIndexPlugin::train_zonemap_index(
        stream,
        test_store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(10)),
    )
    .await
    .unwrap();

    let index = ZoneMapIndex::load(
        test_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert_eq!(index.zones.len(), 1);
    assert_eq!(index.zones[0].null_count, 1, "only the null list row");
    assert_eq!(index.zones[0].nan_count, 0);
    assert!(index.zones[0].min.is_null() && index.zones[0].max.is_null());

    let mut exact_nulls = RowAddrTreeMap::new();
    exact_nulls.insert(0);
    let result = index
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(result, SearchResult::exact(exact_nulls));

    // Bounds are unknown, so value predicates stay conservative.
    assert!(
        index
            .evaluate_zone_against_query(
                &index.zones[0],
                &SargableQuery::Equals(SV::Int32(Some(99)))
            )
            .unwrap()
    );
    assert_eq!(index.value_range(), None);
}

fn f64_zone(
    min: Option<f64>,
    max: Option<f64>,
    null_count: u32,
    nan_count: u32,
    negative_nan_count: Option<u32>,
    length: usize,
) -> ZoneMapStatistics {
    ZoneMapStatistics {
        min: ScalarValue::Float64(min),
        max: ScalarValue::Float64(max),
        null_count,
        nan_count,
        negative_nan_count,
        bound: ZoneBound {
            fragment_id: 0,
            start: 0,
            length,
        },
    }
}

fn range_q(start: Bound<f64>, end: Bound<f64>) -> SargableQuery {
    let conv = |b: Bound<f64>| match b {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(v) => Bound::Included(ScalarValue::Float64(Some(v))),
        Bound::Excluded(v) => Bound::Excluded(ScalarValue::Float64(Some(v))),
    };
    SargableQuery::Range(conv(start), conv(end))
}

/// The NaN sign split makes evaluation exact on both sides; an unknown
/// split stays conservative. The lower side was a live false negative
/// before the split existed.
#[tokio::test]
async fn test_eval_nan_sign_split() {
    use std::ops::Bound;
    let index = train_and_load::<Float64Type>(vec![vec![Some(1.0), Some(2.0)]]).await;
    let eval = |zone: &ZoneMapStatistics, q: &SargableQuery| {
        index.evaluate_zone_against_query(zone, q).unwrap()
    };

    // Only negative NaNs besides the finite range [1, 5].
    let neg = f64_zone(Some(1.0), Some(5.0), 0, 2, Some(2), 10);
    // Only positive NaNs.
    let pos = f64_zone(Some(1.0), Some(5.0), 0, 2, Some(0), 10);
    // Split unknown (legacy / version 0 sources).
    let unknown = f64_zone(Some(1.0), Some(5.0), 0, 2, None, 10);
    // No NaNs at all.
    let finite = f64_zone(Some(1.0), Some(5.0), 0, 0, Some(0), 10);

    // Upper bounds: negative NaNs sort below every finite value.
    let below = range_q(Bound::Unbounded, Bound::Included(0.0));
    assert!(eval(&neg, &below), "negative NaNs match x <= 0");
    assert!(!eval(&pos, &below), "positive NaNs sort above, no match");
    assert!(eval(&unknown, &below), "unknown split stays conservative");
    assert!(!eval(&finite, &below));

    // Lower bounds: positive NaNs sort above every finite value.
    let above = range_q(Bound::Included(10.0), Bound::Unbounded);
    assert!(!eval(&neg, &above));
    assert!(eval(&pos, &above));
    assert!(eval(&unknown, &above));
    assert!(!eval(&finite, &above));

    // Equals and IsIn are sign-aware on NaN targets.
    let eq_pos = SargableQuery::Equals(ScalarValue::Float64(Some(f64::NAN)));
    let eq_neg = SargableQuery::Equals(ScalarValue::Float64(Some(-f64::NAN)));
    assert!(!eval(&neg, &eq_pos));
    assert!(eval(&neg, &eq_neg));
    assert!(eval(&pos, &eq_pos));
    assert!(!eval(&pos, &eq_neg));
    assert!(eval(&unknown, &eq_pos) && eval(&unknown, &eq_neg));
    let in_neg = SargableQuery::IsIn(vec![ScalarValue::Float64(Some(-f64::NAN))]);
    assert!(eval(&neg, &in_neg));
    assert!(!eval(&pos, &in_neg));
}

/// NaN range bounds resolve by sign, including on zones without usable
/// extrema (an all-NaN zone has null bounds).
#[tokio::test]
async fn test_eval_nan_range_bounds() {
    use std::ops::Bound;
    let index = train_and_load::<Float64Type>(vec![vec![Some(1.0), Some(2.0)]]).await;
    let eval = |zone: &ZoneMapStatistics, q: &SargableQuery| {
        index.evaluate_zone_against_query(zone, q).unwrap()
    };
    let finite = f64_zone(Some(1.0), Some(5.0), 0, 0, Some(0), 10);
    let all_neg_nan = f64_zone(None, None, 0, 2, Some(2), 2);

    // x >= -NaN matches every non-null value (was: only NaN-bearing zones).
    assert!(eval(
        &finite,
        &range_q(Bound::Included(-f64::NAN), Bound::Unbounded)
    ));
    // x > -NaN matches every comparable value (was: nothing).
    assert!(eval(
        &finite,
        &range_q(Bound::Excluded(-f64::NAN), Bound::Unbounded)
    ));
    // x > +NaN matches nothing.
    assert!(!eval(
        &finite,
        &range_q(Bound::Excluded(f64::NAN), Bound::Unbounded)
    ));
    // x <= +NaN matches every non-null value; x < +NaN excludes positive NaNs.
    assert!(eval(
        &finite,
        &range_q(Bound::Unbounded, Bound::Included(f64::NAN))
    ));
    assert!(eval(
        &finite,
        &range_q(Bound::Unbounded, Bound::Excluded(f64::NAN))
    ));
    // x < -NaN matches nothing.
    assert!(!eval(
        &finite,
        &range_q(Bound::Unbounded, Bound::Excluded(-f64::NAN))
    ));

    // All-NaN zone (null bounds): x <= -NaN still answers exactly.
    assert!(eval(
        &all_neg_nan,
        &range_q(Bound::Unbounded, Bound::Included(-f64::NAN))
    ));
    assert!(!eval(
        &all_neg_nan,
        &range_q(Bound::Included(f64::NAN), Bound::Unbounded)
    ));
}

/// A version 1 index file round trip keeps the ordinary bounds and the
/// exact NaN sign split, so the loaded zones answer sign-dependent
/// queries exactly.
#[tokio::test]
async fn test_index_file_v1_round_trip_keeps_bounds_and_sign_split() {
    let tmpdir = TempObjDir::default();
    let index = train_and_load_in::<Float64Type>(
        local_store(&tmpdir),
        vec![vec![
            Some(1.0),
            Some(-f64::NAN),
            Some(3.0),
            Some(4.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ]],
    )
    .await;
    assert_eq!(index.version(), ZoneMapIndexVersion::V1);
    // rows_per_zone = 2: [1.0, -NaN], [3.0, 4.0], [+NaN, +inf].
    assert_eq!(index.zones.len(), 3);
    assert_eq!(index.zones[0].min, ScalarValue::Float64(Some(1.0)));
    assert_eq!(index.zones[0].max, ScalarValue::Float64(Some(1.0)));
    assert_eq!(index.zones[0].nan_count, 1);
    assert_eq!(index.zones[0].negative_nan_count, Some(1));
    // The split is known: no positive NaN sorts above 2.0 here, so x > 2
    // is decided by the ordinary bounds alone.
    assert!(
        index
            .evaluate_zone_against_query(
                &index.zones[0],
                &range_q(std::ops::Bound::Unbounded, std::ops::Bound::Included(0.0))
            )
            .unwrap()
    );
    assert!(
        !index
            .evaluate_zone_against_query(
                &index.zones[0],
                &range_q(std::ops::Bound::Excluded(2.0), std::ops::Bound::Unbounded)
            )
            .unwrap()
    );
    assert!(ZoneMapIndex::zone_has_finite_extrema(&index.zones[1]));
    assert_eq!(index.zones[1].negative_nan_count, Some(0));
    // Infinities are ordinary bounds; NaNs are counted, not marked.
    assert_eq!(
        index.zones[2].min,
        ScalarValue::Float64(Some(f64::INFINITY))
    );
    assert_eq!(
        index.zones[2].max,
        ScalarValue::Float64(Some(f64::INFINITY))
    );
    assert_eq!(index.zones[2].nan_count, 1);
    assert_eq!(index.zones[2].negative_nan_count, Some(0));

    // The file carries the version 1 schema.
    let reader = index.store.open_index_file(ZONEMAP_FILENAME).await.unwrap();
    let names: Vec<String> = reader
        .schema()
        .fields
        .iter()
        .map(|f| f.name.clone())
        .collect();
    assert_eq!(
        names,
        [
            "min",
            "max",
            "null_count",
            "nan_count",
            "negative_nan_count",
            "fragment_id",
            "zone_start",
            "zone_length"
        ]
    );
}

/// One version 0 zone row: `(min, max, null_count, nan_count,
/// fragment_id, zone_start, zone_length)`.
type V0ZoneRow = (Option<f64>, Option<f64>, u32, u32, u64, u64, u64);

/// Writes a version 0 Float64 zone map file by hand: seven columns, with
/// the legacy `+NaN` max marker on every zone that holds NaNs.
async fn write_v0_float_zonemap(store: &dyn IndexStore, zones: &[V0ZoneRow]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("min", DataType::Float64, true),
        Field::new("max", DataType::Float64, true),
        Field::new("null_count", DataType::UInt32, false),
        Field::new("nan_count", DataType::UInt32, false),
        Field::new("fragment_id", DataType::UInt64, false),
        Field::new("zone_start", DataType::UInt64, false),
        Field::new("zone_length", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(
                zones.iter().map(|z| z.0).collect::<Vec<_>>(),
            )) as _,
            Arc::new(Float64Array::from(
                zones.iter().map(|z| z.1).collect::<Vec<_>>(),
            )) as _,
            Arc::new(UInt32Array::from_iter_values(zones.iter().map(|z| z.2))) as _,
            Arc::new(UInt32Array::from_iter_values(zones.iter().map(|z| z.3))) as _,
            Arc::new(UInt64Array::from_iter_values(zones.iter().map(|z| z.4))) as _,
            Arc::new(UInt64Array::from_iter_values(zones.iter().map(|z| z.5))) as _,
            Arc::new(UInt64Array::from_iter_values(zones.iter().map(|z| z.6))) as _,
        ],
    )
    .unwrap();
    let mut file_schema = schema.as_ref().clone();
    file_schema
        .metadata
        .insert(ZONEMAP_SIZE_META_KEY.to_string(), "2".to_string());
    let mut writer = store
        .new_index_file(ZONEMAP_FILENAME, Arc::new(file_schema))
        .await
        .unwrap();
    writer.write_record_batch(batch).await.unwrap();
    writer.finish().await.unwrap();
}

// ---- seed reuse into target zones ----

fn seed_of<A: Array + 'static>(
    fragment_id: u64,
    record_rows: u64,
    values: A,
) -> crate::scalar::seed::FragmentSeed {
    use crate::scalar::seed::IndexSeedWriter;
    let values: ArrayRef = Arc::new(values);
    let mut writer = ZoneMapSeedWriter::new("c", record_rows, values.data_type().clone()).unwrap();
    writer.observe_batch(&values).unwrap();
    crate::scalar::seed::FragmentSeed {
        fragment_id,
        physical_rows: values.len() as u64,
        bytes: writer.finish().unwrap().unwrap(),
        metadata_value: format!("0:{record_rows}"),
    }
}

/// The same seed as an older writer produced it: no null positions.
fn without_null_positions(
    seed: &crate::scalar::seed::FragmentSeed,
) -> crate::scalar::seed::FragmentSeed {
    use arrow_ipc::reader::FileReader;
    use arrow_ipc::writer::FileWriter;
    let mut reader = FileReader::try_new(std::io::Cursor::new(seed.bytes.as_ref()), None).unwrap();
    let batch = reader.next().unwrap().unwrap();
    let mut schema = batch.schema().as_ref().clone();
    schema.metadata.remove(SEED_NULL_BITMAP_META_KEY);
    let batch = RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut writer = FileWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    crate::scalar::seed::FragmentSeed {
        fragment_id: seed.fragment_id,
        physical_rows: seed.physical_rows,
        bytes: bytes::Bytes::from(buf.into_inner()),
        metadata_value: seed.metadata_value.clone(),
    }
}

fn harvested(seed: crate::scalar::seed::FragmentSeed, physical_rows: u64) -> HarvestedZoneSeed {
    HarvestedZoneSeed {
        seed,
        physical_rows,
    }
}

fn int_values(n: i32, null_every: i32) -> Int32Array {
    Int32Array::from_iter((0..n).map(|i| (null_every == 0 || i % null_every != 0).then_some(i * 3)))
}

fn zone_bounds(reused: &ReusedZones) -> Vec<(u64, u64, usize)> {
    let mut v: Vec<(u64, u64, usize)> = reused
        .zones
        .values()
        .map(|z| (z.bound.fragment_id, z.bound.start, z.bound.length))
        .collect();
    v.sort();
    v
}

#[test]
fn test_reuse_places_whole_records_into_their_target_zone() {
    // Ten rows in records of four: [0,4) [4,8) [8,10); nulls at rows 0, 5.
    let seed = seed_of(7, 4, int_values(10, 5));

    // Target 4: every record is its own zone.
    let mut reused = ReusedZones::new(true);
    let outcome = reused
        .reuse(&harvested(seed.clone(), 10), 4, &DataType::Int32, true)
        .unwrap();
    assert_eq!(outcome, SeedOutcome::Complete);
    assert_eq!(zone_bounds(&reused), vec![(7, 0, 4), (7, 4, 4), (7, 8, 2)]);
    assert_eq!(reused.reused_records, 3);
    assert_eq!(reused.null_rows.len(), Some(2));

    // Target 8: records 0 and 1 fold into zone 0, record 2 is zone 1.
    let mut reused = ReusedZones::new(false);
    let outcome = reused
        .reuse(&harvested(seed.clone(), 10), 8, &DataType::Int32, true)
        .unwrap();
    assert_eq!(outcome, SeedOutcome::Complete);
    assert_eq!(zone_bounds(&reused), vec![(7, 0, 8), (7, 8, 2)]);
    let zone0 = &reused.zones[&(7, 0)];
    assert_eq!(zone0.min, ScalarValue::Int32(Some(3)));
    assert_eq!(zone0.max, ScalarValue::Int32(Some(21)));
    assert_eq!(zone0.null_count, 2);
    assert_eq!(reused.zones[&(7, 1)].min, ScalarValue::Int32(Some(24)));

    // Target 10: everything is zone 0.
    let mut reused = ReusedZones::new(false);
    reused
        .reuse(&harvested(seed.clone(), 10), 10, &DataType::Int32, true)
        .unwrap();
    assert_eq!(zone_bounds(&reused), vec![(7, 0, 10)]);
    assert_eq!(reused.zones[&(7, 0)].null_count, 2);

    // Target 6: [0,4) fits zone 0; [4,8) crosses 6; [8,10) fits zone 1.
    // Partial: rows [4,8) are left to the scan, and the positions inside
    // them (row 5) with them.
    let mut reused = ReusedZones::new(true);
    let outcome = reused
        .reuse(&harvested(seed.clone(), 10), 6, &DataType::Int32, true)
        .unwrap();
    assert_eq!(outcome, SeedOutcome::Partial(vec![4..8]));
    assert_eq!(zone_bounds(&reused), vec![(7, 0, 6), (7, 6, 4)]);
    assert_eq!(reused.zones[&(7, 0)].null_count, 1);
    assert_eq!(reused.zones[&(7, 0)].max, ScalarValue::Int32(Some(9)));
    assert_eq!(reused.zones[&(7, 1)].min, ScalarValue::Int32(Some(24)));
    assert!(reused.null_rows.contains(7 << 32));
    assert!(!reused.null_rows.contains((7 << 32) | 5));
    assert_eq!(reused.partial_fragments, vec![7]);

    // The same without partial reuse allowed: nothing kept.
    let mut reused = ReusedZones::new(true);
    let outcome = reused
        .reuse(&harvested(seed.clone(), 10), 6, &DataType::Int32, false)
        .unwrap();
    assert_eq!(outcome, SeedOutcome::Rejected);
    assert!(reused.zones.is_empty());
    assert_eq!(reused.rejected_fragments, vec![7]);

    // Target 3: no record fits, the fragment is scanned whole.
    let mut reused = ReusedZones::new(false);
    let outcome = reused
        .reuse(&harvested(seed, 10), 3, &DataType::Int32, true)
        .unwrap();
    assert_eq!(outcome, SeedOutcome::Rejected);
}

/// A narrow column's partial reuse is capped: once more than half of a
/// fragment's rows sit in records that cross a target boundary, the fragment
/// is scanned whole instead of in scattered intervals. A wide column keeps
/// partial reuse at any fraction.
#[test]
fn test_reuse_caps_rescan_for_narrow_columns() {
    // Ten rows in records of three: [0,3) [3,6) [6,9) [9,10).
    let seed = seed_of(7, 3, int_values(10, 0));

    // Target 4: [3,6) crosses 4 and [6,9) crosses 8, six of ten rows.
    let mut reused = ReusedZones::new(false);
    let outcome = reused
        .reuse(&harvested(seed.clone(), 10), 4, &DataType::Int32, true)
        .unwrap();
    assert_eq!(outcome, SeedOutcome::Rejected);
    assert!(reused.zones.is_empty());
    assert_eq!(reused.rejected_fragments, vec![7]);

    // Target 5: only [3,6) crosses, three of ten rows.
    let mut reused = ReusedZones::new(false);
    let outcome = reused
        .reuse(&harvested(seed, 10), 5, &DataType::Int32, true)
        .unwrap();
    assert_eq!(outcome, SeedOutcome::Partial(vec![3..6]));
    assert_eq!(reused.partial_fragments, vec![7]);

    // The same six of ten rows on a string column stay partial.
    let seed = seed_of(
        7,
        3,
        arrow_array::StringArray::from_iter_values((0..10).map(|i| format!("v{i:02}"))),
    );
    let mut reused = ReusedZones::new(false);
    let outcome = reused
        .reuse(&harvested(seed, 10), 4, &DataType::Utf8, true)
        .unwrap();
    assert_eq!(outcome, SeedOutcome::Partial(vec![3..6, 6..9]));
}

/// Non-divisible sizes: the boundary phase advances by `T mod s` per
/// zone, so the crossing record and the number of whole records per
/// zone change from zone to zone. Checked against a brute-force
/// intersection over synthetic records (no rows allocated).
#[test]
fn test_place_records_cycles_through_boundary_phases() {
    fn records(record_rows: u64, physical_rows: u64) -> Vec<ZoneMapStatistics> {
        let mut out = Vec::new();
        let mut start = 0;
        while start < physical_rows {
            let length = record_rows.min(physical_rows - start);
            out.push(ZoneMapStatistics {
                min: ScalarValue::Int32(Some(0)),
                max: ScalarValue::Int32(Some(1)),
                null_count: 0,
                nan_count: 0,
                negative_nan_count: Some(0),
                bound: ZoneBound {
                    fragment_id: 0,
                    start,
                    length: length as usize,
                },
            });
            start += length;
        }
        out
    }
    for (record_rows, target, physical_rows) in [
        (8192u64, 1_000_000u64, 16_000_000u64),
        (8192, 10_000, 200_000),
        (8192, 8192 * 3, 100_000),
        (8192, 1000, 50_000),
        (10, 25, 103),
    ] {
        let recs = records(record_rows, physical_rows);
        let placement = place_records(&recs, target, physical_rows);
        // Brute force: a record fits iff all its rows share a zone index.
        let mut expect_fitted = Vec::new();
        let mut expect_uncovered = Vec::new();
        for r in &recs {
            let start = r.bound.start;
            let end = start + r.bound.length as u64;
            if (start..end)
                .map(|o| o / target)
                .all(|k| k == start / target)
            {
                expect_fitted.push((start / target, start));
            } else {
                expect_uncovered.push(start..end);
            }
        }
        let fitted: Vec<(u64, u64)> = placement
            .fitted
            .iter()
            .map(|(k, r)| (*k, r.bound.start))
            .collect();
        assert_eq!(fitted, expect_fitted, "{record_rows} -> {target}");
        assert_eq!(
            placement.uncovered, expect_uncovered,
            "{record_rows} -> {target}"
        );
        // Fitted records plus uncovered intervals tile the fragment.
        let mut covered: u64 = placement
            .fitted
            .iter()
            .map(|(_, r)| r.bound.length as u64)
            .sum();
        covered += placement
            .uncovered
            .iter()
            .map(|i| i.end - i.start)
            .sum::<u64>();
        assert_eq!(covered, physical_rows);
    }
    // The phase for 8192 -> 1,000,000: the record crossing the first
    // boundary is [999424, 1007616), the one crossing the second is
    // [1998848, 2007040); the last record [2998272, 3000000) fits zone 2.
    let recs = records(8192, 3_000_000);
    let placement = place_records(&recs, 1_000_000, 3_000_000);
    assert_eq!(
        placement.uncovered,
        vec![999_424..1_007_616, 1_998_848..2_007_040]
    );
    let zone1_whole = placement.fitted.iter().filter(|(k, _)| *k == 1).count();
    assert_eq!(zone1_whole, 121);
    assert_eq!(
        placement.fitted.iter().filter(|(k, _)| *k == 0).count(),
        122
    );
}

#[test]
fn test_reuse_validates_coverage_type_and_positions() {
    let seed = seed_of(1, 4, int_values(10, 0));

    // The file holds more rows than the seed describes.
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(seed.clone(), 11), 100, &DataType::Int32, true)
            .unwrap(),
        SeedOutcome::Rejected
    );

    // The column type changed since the seed was written.
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(seed.clone(), 10), 100, &DataType::Int64, true)
            .unwrap(),
        SeedOutcome::Rejected
    );

    // An unparsable locator.
    let mut bad = seed;
    bad.metadata_value = "0".to_string();
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(bad, 10), 100, &DataType::Int32, true)
            .unwrap(),
        SeedOutcome::Rejected
    );

    // A seed without positions is fine when the index keeps none and is
    // rejected (scanned) when the index keeps them.
    let old = without_null_positions(&seed_of(2, 4, int_values(10, 3)));
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(old.clone(), 10), 100, &DataType::Int32, true)
            .unwrap(),
        SeedOutcome::Complete
    );
    let mut reused = ReusedZones::new(true);
    assert_eq!(
        reused
            .reuse(&harvested(old, 10), 100, &DataType::Int32, true)
            .unwrap(),
        SeedOutcome::Rejected
    );
}

#[test]
fn test_reuse_rejects_missing_bounds_but_accepts_all_null_records() {
    use crate::scalar::seed::FragmentSeed;

    fn seed_from_records(records: &[ZoneMapStatistics]) -> FragmentSeed {
        use arrow_ipc::writer::FileWriter;
        let batch = ZoneMapSeedWriter::seed_batch_from_zones(records, &DataType::Int32).unwrap();
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut writer = FileWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        FragmentSeed {
            fragment_id: 3,
            physical_rows: records.iter().map(|r| r.bound.length as u64).sum(),
            bytes: bytes::Bytes::from(buf.into_inner()),
            metadata_value: "0:4".to_string(),
        }
    }
    let record =
        |min: Option<i32>, max: Option<i32>, null_count: u32, start: u64, length: usize| {
            ZoneMapStatistics {
                min: ScalarValue::Int32(min),
                max: ScalarValue::Int32(max),
                null_count,
                nan_count: 0,
                negative_nan_count: Some(0),
                bound: ZoneBound {
                    fragment_id: 3,
                    start,
                    length,
                },
            }
        };

    // Comparable values without bounds: scanned.
    let seed = seed_from_records(&[record(None, None, 1, 0, 4)]);
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(seed, 4), 8, &DataType::Int32, true)
            .unwrap(),
        SeedOutcome::Rejected
    );

    // All-null records legitimately have no bounds and fold with others.
    let seed = seed_from_records(&[
        record(None, None, 4, 0, 4),
        record(Some(5), Some(9), 0, 4, 2),
    ]);
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(seed, 6), 8, &DataType::Int32, true)
            .unwrap(),
        SeedOutcome::Complete
    );
    let zone = &reused.zones[&(3, 0)];
    assert_eq!(zone.min, ScalarValue::Int32(Some(5)));
    assert_eq!(zone.max, ScalarValue::Int32(Some(9)));
    assert_eq!(zone.null_count, 4);
    assert_eq!(zone.bound.length, 6);

    // Counts beyond the rows are rejected.
    let seed = seed_from_records(&[record(Some(1), Some(2), 5, 0, 4)]);
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(seed, 6), 8, &DataType::Int32, true)
            .unwrap(),
        SeedOutcome::Rejected
    );
    // Short records before the last one are a binary-copy carry's
    // concatenation; they are placed by their own lengths.
    let seed = seed_from_records(&[
        record(Some(1), Some(2), 0, 0, 3),
        record(Some(3), Some(4), 0, 3, 3),
    ]);
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(seed, 6), 8, &DataType::Int32, true)
            .unwrap(),
        SeedOutcome::Complete
    );
    let zone = &reused.zones[&(3, 0)];
    assert_eq!(zone.min, ScalarValue::Int32(Some(1)));
    assert_eq!(zone.max, ScalarValue::Int32(Some(4)));
}

/// Reused records keep the version 1 contract: infinities are ordinary
/// bounds, NaNs are counted with their sign and never become a bound.
#[test]
fn test_reuse_keeps_ordinary_infinities_and_sign_split() {
    use arrow_array::Float32Array;
    // Records of four: [-inf, 1, 2, 3], [4, NaN, 6, inf], [7, 8].
    let values = Float32Array::from(vec![
        Some(f32::NEG_INFINITY),
        Some(1.0),
        Some(2.0),
        Some(3.0),
        Some(4.0),
        Some(f32::NAN),
        Some(6.0),
        Some(f32::INFINITY),
        Some(7.0),
        Some(8.0),
    ]);
    let seed = seed_of(4, 4, values);
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(seed, 10), 10, &DataType::Float32, true)
            .unwrap(),
        SeedOutcome::Complete
    );
    let zone = &reused.zones[&(4, 0)];
    assert_eq!(zone.min, ScalarValue::Float32(Some(f32::NEG_INFINITY)));
    assert_eq!(zone.max, ScalarValue::Float32(Some(f32::INFINITY)));
    assert_eq!(zone.nan_count, 1);
    assert_eq!(zone.negative_nan_count, Some(0));
    assert_eq!(zone.null_count, 0);
    assert_eq!(zone.bound.length, 10);
}

/// A version 0 seed record with NaNs cannot supply a version 1 zone's
/// ordinary maximum or sign split, so its rows are scanned; the other
/// records of the same seed are still reused with a zero split.
#[test]
fn test_reuse_scans_version_0_records_with_nans() {
    use crate::scalar::seed::FragmentSeed;
    use arrow_array::Float32Array;
    use arrow_ipc::writer::FileWriter;
    // A version 0 writer's records of two over [1, 2, NaN, 4, 5, 6]: the
    // middle record carries the +NaN max marker, no sign split.
    let schema = Arc::new(Schema::new(vec![
        Field::new("min", DataType::Float32, true),
        Field::new("max", DataType::Float32, true),
        Field::new("null_count", DataType::UInt32, false),
        Field::new("nan_count", DataType::UInt32, false),
        Field::new("zone_length", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float32Array::from(vec![Some(1.0), Some(4.0), Some(5.0)])),
            Arc::new(Float32Array::from(vec![
                Some(2.0),
                Some(f32::NAN),
                Some(6.0),
            ])),
            Arc::new(UInt32Array::from(vec![0, 0, 0])),
            Arc::new(UInt32Array::from(vec![0, 1, 0])),
            Arc::new(UInt64Array::from(vec![2u64, 2, 2])),
        ],
    )
    .unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut w = FileWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();
    }
    let v0 = FragmentSeed {
        fragment_id: 5,
        physical_rows: 6,
        bytes: bytes::Bytes::from(buf.into_inner()),
        metadata_value: "0:2".to_string(),
    };
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(v0.clone(), 6), 2, &DataType::Float32, true)
            .unwrap(),
        SeedOutcome::Partial(vec![2..4])
    );
    let zone = &reused.zones[&(5, 0)];
    assert_eq!(zone.max, ScalarValue::Float32(Some(2.0)));
    assert_eq!(zone.negative_nan_count, Some(0));
    assert!(!reused.zones.contains_key(&(5, 1)));
    assert_eq!(reused.zones[&(5, 2)].min, ScalarValue::Float32(Some(5.0)));
    // Without partial reuse the whole fragment is scanned.
    let mut reused = ReusedZones::new(false);
    assert_eq!(
        reused
            .reuse(&harvested(v0, 6), 2, &DataType::Float32, false)
            .unwrap(),
        SeedOutcome::Rejected
    );
}

/// Build a segment from batches, one fragment per batch (fragment id =
/// batch position), with the aligned trainer knowing each fragment's
/// physical row count.
async fn build_segment_from_batches(
    store: &dyn IndexStore,
    params: ZoneMapIndexBuilderParams,
    fragments: Vec<Vec<Option<i32>>>,
) -> CreatedIndex {
    let schema = Arc::new(Schema::new(vec![Field::new(
        VALUE_COLUMN_NAME,
        DataType::Int32,
        true,
    )]));
    let physical_rows: HashMap<u64, u64> = fragments
        .iter()
        .enumerate()
        .map(|(i, rows)| (i as u64, rows.len() as u64))
        .collect();
    let batches: Vec<RecordBatch> = fragments
        .into_iter()
        .map(|vals| {
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vals))]).unwrap()
        })
        .collect();
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter(batches.into_iter().map(Ok)),
    ));
    build_zonemap_segment(
        ZoneMapSegmentBase::New(params),
        &DataType::Int32,
        ReusedZones::default(),
        Some(add_row_addr(stream)),
        &physical_rows,
        &[],
        store,
    )
    .await
    .unwrap()
}

fn local_store(dir: &TempObjDir) -> Arc<LanceIndexStore> {
    Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        (**dir).clone(),
        Arc::new(LanceCache::no_cache()),
    ))
}

/// A version 0 file is read under the version 0 contract: the NaN marker
/// stays in `max` as data, the sign split is unknown for zones with NaNs
/// and provably zero without them, and the zones stay conservative.
#[tokio::test]
async fn test_index_file_v0_read_keeps_legacy_contract() {
    let tmpdir = TempObjDir::default();
    let store = local_store(&tmpdir);
    write_v0_float_zonemap(
        store.as_ref(),
        &[
            (Some(1.0), Some(f64::NAN), 0, 1, 0, 0, 2),
            (Some(3.0), Some(4.0), 0, 0, 0, 2, 2),
        ],
    )
    .await;
    let index = ZoneMapIndex::load(
        store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V0,
    )
    .await
    .unwrap();
    assert_eq!(index.version(), ZoneMapIndexVersion::V0);
    assert_eq!(index.rows_per_zone, 2);
    assert!(ZoneMapIndex::scalar_is_nan(&index.zones[0].max));
    assert_eq!(index.zones[0].negative_nan_count, None);
    assert_eq!(index.zones[1].negative_nan_count, Some(0));
    // Unknown split: both signs stay possible, so x > 2 keeps the zone
    // even though its hidden ordinary max is 1.0.
    assert!(
        index
            .evaluate_zone_against_query(
                &index.zones[0],
                &range_q(std::ops::Bound::Excluded(2.0), std::ops::Bound::Unbounded)
            )
            .unwrap()
    );
    assert!(
        index
            .evaluate_zone_against_query(
                &index.zones[0],
                &range_q(std::ops::Bound::Unbounded, std::ops::Bound::Included(0.0))
            )
            .unwrap()
    );
    // The same file is not a valid version 1 file.
    let err = ZoneMapIndex::load(
        store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("negative_nan_count"),
        "unexpected error: {err}"
    );
}

/// Version 1 files are validated on load: an impossible sign split or a
/// NaN bound is a corrupt file, not a conservative zone.
#[tokio::test]
async fn test_index_file_v1_rejects_invalid_zones() {
    async fn write_v1(store: &dyn IndexStore, max: f64, nan_count: u32, negative_nan_count: u32) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("min", DataType::Float64, true),
            Field::new("max", DataType::Float64, true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("nan_count", DataType::UInt32, false),
            Field::new("negative_nan_count", DataType::UInt32, false),
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("zone_start", DataType::UInt64, false),
            Field::new("zone_length", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(vec![Some(1.0)])) as _,
                Arc::new(Float64Array::from(vec![Some(max)])) as _,
                Arc::new(UInt32Array::from(vec![0u32])) as _,
                Arc::new(UInt32Array::from(vec![nan_count])) as _,
                Arc::new(UInt32Array::from(vec![negative_nan_count])) as _,
                Arc::new(UInt64Array::from(vec![0u64])) as _,
                Arc::new(UInt64Array::from(vec![0u64])) as _,
                Arc::new(UInt64Array::from(vec![4u64])) as _,
            ],
        )
        .unwrap();
        let mut file_schema = schema.as_ref().clone();
        file_schema
            .metadata
            .insert(ZONEMAP_SIZE_META_KEY.to_string(), "4".to_string());
        let mut writer = store
            .new_index_file(ZONEMAP_FILENAME, Arc::new(file_schema))
            .await
            .unwrap();
        writer.write_record_batch(batch).await.unwrap();
        writer.finish().await.unwrap();
    }
    for (max, nan_count, negative, needle) in [
        (2.0, 1, 2, "negative_nan_count"),
        (f64::NAN, 1, 0, "NaN bound"),
        (2.0, 5, 0, "above its length"),
    ] {
        let tmpdir = TempObjDir::default();
        let store = local_store(&tmpdir);
        write_v1(store.as_ref(), max, nan_count, negative).await;
        let err = ZoneMapIndex::load(
            store,
            None,
            &LanceCache::no_cache(),
            false,
            ZoneMapIndexVersion::V1,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains(needle), "unexpected error: {err}");
    }
}

/// Versions this build does not know are refused, both by the enum and
/// by the plugin's load entries; nothing is interpreted under the
/// version 0 or version 1 contract.
#[tokio::test]
async fn test_unsupported_index_version_is_refused() {
    assert_eq!(
        ZoneMapIndexVersion::from_metadata(0).unwrap(),
        ZoneMapIndexVersion::V0
    );
    assert_eq!(
        ZoneMapIndexVersion::from_metadata(1).unwrap(),
        ZoneMapIndexVersion::V1
    );
    assert_eq!(ZoneMapIndexPlugin.version(), 1);
    let err = ZoneMapIndexVersion::from_metadata(2).unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");

    let tmpdir = TempObjDir::default();
    let index =
        train_and_load_in::<Float64Type>(local_store(&tmpdir), vec![vec![Some(1.0), Some(2.0)]])
            .await;
    let details = make_zone_map_index_details(2, false, true);
    let err = ZoneMapIndexPlugin
        .load_index(
            index.store.clone(),
            &details,
            2,
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
    let err = ZoneMapIndexPlugin
        .load_index_with_remapping(
            index.store.clone(),
            &details,
            2,
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
    // Version 1 loads through the plugin.
    let loaded = ZoneMapIndexPlugin
        .load_index(
            index.store.clone(),
            &details,
            1,
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();
    assert_eq!(
        loaded
            .as_any()
            .downcast_ref::<ZoneMapIndex>()
            .unwrap()
            .version(),
        ZoneMapIndexVersion::V1
    );
}

/// A direct merge keeps the sources' file version: version 0 sources
/// produce a version 0 file (marker and unknown split preserved),
/// version 1 sources a version 1 file, and mixed versions are refused
/// before anything is written.
#[tokio::test]
async fn test_merge_keeps_source_version_and_refuses_mixed_versions() {
    let v1_a = train_and_load::<Float64Type>(vec![vec![Some(1.0), Some(-f64::NAN)]]).await;
    let mut v1_b = train_and_load::<Float64Type>(vec![vec![Some(3.0), Some(4.0)]]).await;
    for zone in &mut Arc::get_mut(&mut v1_b).unwrap().zones {
        zone.bound.fragment_id = 1;
    }
    let tmpdir_a = TempObjDir::default();
    let store_a = local_store(&tmpdir_a);
    write_v0_float_zonemap(
        store_a.as_ref(),
        &[(Some(1.0), Some(f64::NAN), 0, 1, 0, 0, 2)],
    )
    .await;
    let tmpdir_b = TempObjDir::default();
    let store_b = local_store(&tmpdir_b);
    write_v0_float_zonemap(store_b.as_ref(), &[(Some(3.0), Some(4.0), 0, 0, 1, 0, 2)]).await;
    let mut v0 = Vec::new();
    for store in [store_a, store_b] {
        v0.push(
            ZoneMapIndex::load(
                store,
                None,
                &LanceCache::no_cache(),
                false,
                ZoneMapIndexVersion::V0,
            )
            .await
            .unwrap(),
        );
    }
    let all_frags = RoaringBitmap::from_iter([0u32, 1]);

    // Version 0 + version 0 -> version 0.
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    let created = merge_zonemap_indices(
        &[v0[0].as_ref(), v0[1].as_ref()],
        dest_store.as_ref(),
        &all_frags,
    )
    .await
    .unwrap();
    assert_eq!(created.index_version, 0);
    let merged = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V0,
    )
    .await
    .unwrap();
    assert_eq!(merged.zones.len(), 2);
    assert!(ZoneMapIndex::scalar_is_nan(&merged.zones[0].max));
    assert_eq!(merged.zones[0].negative_nan_count, None);
    assert_eq!(merged.zones[1].negative_nan_count, Some(0));

    // Version 1 + version 1 -> version 1.
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    let created = merge_zonemap_indices(
        &[v1_a.as_ref(), v1_b.as_ref()],
        dest_store.as_ref(),
        &all_frags,
    )
    .await
    .unwrap();
    assert_eq!(created.index_version, 1);
    let merged = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert_eq!(merged.zones.len(), 2);
    assert_eq!(merged.zones[0].max, ScalarValue::Float64(Some(1.0)));
    assert_eq!(merged.zones[0].negative_nan_count, Some(1));

    // Mixed versions are refused before any write, in either order.
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    for sources in [
        vec![v1_a.as_ref(), v0[1].as_ref()],
        vec![v0[0].as_ref(), v1_b.as_ref()],
    ] {
        let err = merge_zonemap_indices(&sources, dest_store.as_ref(), &all_frags)
            .await
            .err()
            .expect("merging mixed ZoneMap versions must be refused");
        assert!(matches!(err, Error::NotSupported { .. }), "{err}");
        assert!(
            dest_store.open_index_file(ZONEMAP_FILENAME).await.is_err(),
            "a refused merge must not write an index file"
        );
    }
}

/// Builds a version 1 Float64 index with `rows_per_zone` into `store`.
async fn train_float_index_in(
    store: Arc<LanceIndexStore>,
    rows_per_zone: u64,
    values: Vec<Option<f64>>,
) -> Arc<ZoneMapIndex> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        VALUE_COLUMN_NAME,
        DataType::Float64,
        true,
    )]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Float64Array::from(values))]).unwrap();
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::once(std::future::ready(Ok(batch))),
    ));
    ZoneMapIndexPlugin::train_zonemap_index(
        add_row_addr(stream),
        store.as_ref(),
        Some(ZoneMapIndexBuilderParams::new(rows_per_zone)),
    )
    .await
    .unwrap();
    ZoneMapIndex::load(
        store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap()
}

/// The planner rewrites an ordered float comparison into a total-order
/// range plus `OR other < -inf`, which exists only to retrieve negative
/// NaNs. Pruning that range must never drop a zone that really holds one:
/// the rows would be missing from the answer. Version 1 records the NaN
/// sign split, so it may prune only the zones whose NaNs are all positive.
#[tokio::test]
async fn test_negative_nan_range_keeps_zones_holding_negative_nans() {
    // Zone 0: a negative NaN among finite values. Zone 1: positive NaNs
    // only. Zone 2: finite only.
    let tmpdir = TempObjDir::default();
    let index = train_float_index_in(
        local_store(&tmpdir),
        3,
        vec![
            Some(-f64::NAN),
            Some(1.0),
            Some(2.0),
            Some(f64::NAN),
            Some(f64::NAN),
            Some(3.0),
            Some(4.0),
            Some(5.0),
            Some(6.0),
        ],
    )
    .await;
    assert_eq!(index.zones.len(), 3);
    assert_eq!(index.zones[0].negative_nan_count, Some(1));
    assert_eq!(index.zones[1].negative_nan_count, Some(0));
    assert_eq!(index.zones[2].nan_count, 0);

    let query = SargableQuery::Range(
        Bound::Unbounded,
        Bound::Excluded(ScalarValue::Float64(Some(f64::NEG_INFINITY))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(
        result,
        SearchResult::at_most(RowAddrTreeMap::from_iter(0..3)),
        "only the zone holding a negative NaN may survive, and it must"
    );
}

/// Version 0 has no NaN sign split to consult, so the same range must keep
/// every NaN-bearing zone as a candidate even when all of its NaNs are in
/// fact positive. Pruning it there would drop rows the planner asked for.
#[tokio::test]
async fn test_negative_nan_range_is_conservative_in_version_0() {
    let tmpdir = TempObjDir::default();
    let mut index = train_float_index_in(
        local_store(&tmpdir),
        3,
        vec![
            Some(f64::NAN),
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0),
        ],
    )
    .await;
    let index_mut = Arc::get_mut(&mut index).unwrap();
    index_mut.version = ZoneMapIndexVersion::V0;
    for zone in index_mut.zones.iter_mut() {
        zone.negative_nan_count = None;
    }

    let query = SargableQuery::Range(
        Bound::Unbounded,
        Bound::Excluded(ScalarValue::Float64(Some(f64::NEG_INFINITY))),
    );
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(
        result,
        SearchResult::at_most(RowAddrTreeMap::from_iter(0..3)),
        "without a sign split the NaN-bearing zone stays a candidate"
    );
}

/// Remapping copies a zone's counts into each destination run. A version
/// 1 file must hold exact counts, so a merge persists a narrowed run only
/// when the counts provably survive (no NaNs, nulls recomputed from an
/// exact bitmap or absent); otherwise it is refused before any write
/// rather than producing a file its own loader rejects.
#[tokio::test]
async fn test_merge_refuses_inexact_remapped_zones_in_version_1() {
    let new_row = 3_u64 << 32;
    // [+NaN, +NaN, 5.0]: keeping only offset 2 would carry nan_count 2
    // into a run of length 1.
    let tmpdir = TempObjDir::default();
    let mut source = train_float_index_in(
        local_store(&tmpdir),
        3,
        vec![Some(f64::NAN), Some(f64::NAN), Some(5.0)],
    )
    .await;
    assert_eq!(source.zones[0].nan_count, 2);
    let source_mut = Arc::get_mut(&mut source).unwrap();
    source_mut.fri = Some(Arc::new(TestRemapper::new([(2, new_row)])));
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    let err = merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([3]),
    )
    .await
    .err()
    .expect("a narrowed zone with NaNs must not be persisted as version 1");
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
    assert!(dest_store.open_index_file(ZONEMAP_FILENAME).await.is_err());

    // [null, 5.0, 6.0] without an exact bitmap: the conservative null
    // count of the narrowed run is not exact either.
    let tmpdir = TempObjDir::default();
    let mut source =
        train_float_index_in(local_store(&tmpdir), 3, vec![None, Some(5.0), Some(6.0)]).await;
    let source_mut = Arc::get_mut(&mut source).unwrap();
    source_mut.null_rows = None;
    source_mut.fri = Some(Arc::new(TestRemapper::new([
        (1, new_row),
        (2, new_row + 1),
    ])));
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    let err = merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([3]),
    )
    .await
    .err()
    .expect("a narrowed mixed-null zone without a bitmap must not be persisted");
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");

    // The same zone with its exact bitmap: the run's null count is
    // recomputed, so the merge proceeds and the file loads as version 1.
    let tmpdir = TempObjDir::default();
    let mut source =
        train_float_index_in(local_store(&tmpdir), 3, vec![None, Some(5.0), Some(6.0)]).await;
    assert!(source.null_rows.is_some());
    let source_mut = Arc::get_mut(&mut source).unwrap();
    source_mut.fri = Some(Arc::new(TestRemapper::new([
        (1, new_row),
        (2, new_row + 1),
    ])));
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([3]),
    )
    .await
    .unwrap();
    let merged = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert_eq!(merged.zones.len(), 1);
    assert_eq!(merged.zones[0].bound.length, 2);
    assert_eq!(merged.zones[0].null_count, 0);
    assert_eq!(merged.zones[0].nan_count, 0);
    assert_eq!(merged.zones[0].negative_nan_count, Some(0));

    // An all-null zone narrowed without a bitmap: every surviving row is
    // a null, so the run's count is its own length and the merge is
    // persisted as version 1 rather than forced into a rebuild.
    let tmpdir = TempObjDir::default();
    let mut source = train_float_index_in(local_store(&tmpdir), 3, vec![None, None, None]).await;
    assert_eq!(source.zones[0].null_count, 3);
    let source_mut = Arc::get_mut(&mut source).unwrap();
    source_mut.null_rows = None;
    source_mut.fri = Some(Arc::new(TestRemapper::new([
        (1, new_row),
        (2, new_row + 1),
    ])));
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([3]),
    )
    .await
    .expect("a narrowed all-null zone keeps an exact null count");
    let merged = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert_eq!(merged.zones.len(), 1);
    assert_eq!(merged.zones[0].bound.length, 2);
    assert_eq!(
        merged.zones[0].null_count, 2,
        "the run's null count is its own length"
    );
    assert_eq!(merged.zones[0].nan_count, 0);
    assert_eq!(merged.zones[0].negative_nan_count, Some(0));
    assert!(merged.zones[0].min.is_null() && merged.zones[0].max.is_null());
    assert_eq!(
        merged
            .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
            .await
            .unwrap(),
        SearchResult::at_most(RowAddrTreeMap::from_iter(new_row..new_row + 2)),
        "the narrowed zone still answers IS NULL for its rows"
    );

    // A run that keeps every row of its zone keeps exact counts.
    let tmpdir = TempObjDir::default();
    let mut source = train_float_index_in(
        local_store(&tmpdir),
        3,
        vec![Some(f64::NAN), Some(-f64::NAN), Some(5.0)],
    )
    .await;
    let source_mut = Arc::get_mut(&mut source).unwrap();
    source_mut.fri = Some(Arc::new(TestRemapper::new([
        (0, new_row),
        (1, new_row + 1),
        (2, new_row + 2),
    ])));
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([3]),
    )
    .await
    .unwrap();
    let merged = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert_eq!(merged.zones[0].bound.length, 3);
    assert_eq!(merged.zones[0].nan_count, 2);
    assert_eq!(merged.zones[0].negative_nan_count, Some(1));
}

/// Destination runs outside the output's fragments are dropped before
/// they are built or checked: a narrowed NaN-bearing run and a mixed-null
/// singleton without a bitmap, both excluded, do not block the merge of
/// the retained exact run, and the output holds only that coverage.
#[tokio::test]
async fn test_merge_ignores_inexact_runs_outside_the_fragment_filter() {
    let new_row = |fragment: u64, offset: u64| (fragment << 32) + offset;
    // Zones of three: [+NaN, +NaN, 5.0], [null, 7.0, 8.0], [1.0, 2.0, 3.0].
    let tmpdir = TempObjDir::default();
    let mut source = train_float_index_in(
        local_store(&tmpdir),
        3,
        vec![
            Some(f64::NAN),
            Some(f64::NAN),
            Some(5.0),
            None,
            Some(7.0),
            Some(8.0),
            Some(1.0),
            Some(2.0),
            Some(3.0),
        ],
    )
    .await;
    let source_mut = Arc::get_mut(&mut source).unwrap();
    source_mut.null_rows = None;
    // Offset 2 alone to fragment 3 (narrowed, NaNs lost), offset 4 alone
    // to fragment 5 (mixed-null singleton without positions), and the
    // whole third zone to fragment 4.
    source_mut.fri = Some(Arc::new(TestRemapper::new([
        (2, new_row(3, 0)),
        (4, new_row(5, 0)),
        (6, new_row(4, 0)),
        (7, new_row(4, 1)),
        (8, new_row(4, 2)),
    ])));
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    let created = merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([4u32]),
    )
    .await
    .unwrap();
    assert_eq!(created.index_version, 1);
    let merged = ZoneMapIndex::load(
        dest_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert_eq!(merged.zones.len(), 1);
    assert_eq!(merged.zones[0].bound.fragment_id, 4);
    assert_eq!(merged.zones[0].bound.length, 3);
    assert_eq!(merged.zones[0].min, ScalarValue::Float64(Some(1.0)));
    assert_eq!(merged.zones[0].max, ScalarValue::Float64(Some(3.0)));
    assert_eq!(merged.zones[0].negative_nan_count, Some(0));
    // The same excluded runs still refuse the merge once they are kept.
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    let err = merge_zonemap_indices(
        &[source.as_ref()],
        dest_store.as_ref(),
        &RoaringBitmap::from_iter([3u32, 4]),
    )
    .await
    .err()
    .expect("a retained narrowed NaN run must be refused");
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
}

/// Optional version 0 statistics are checked before they are admitted:
/// a record whose counts do not fit its length, or a seed that does not
/// cover its fragment, makes the plugin entry fall back to a scan
/// instead of failing at the version 1 writer.
#[tokio::test]
async fn test_update_with_seeds_scans_invalid_version_0_records() {
    use crate::scalar::seed::FragmentSeed;
    use arrow_ipc::writer::FileWriter;
    /// `(min, max, null_count, nan_count, zone_length)` of a version 0 record.
    type V0SeedRecord = (Option<f64>, Option<f64>, u32, u32, u64);
    fn v0_seed(fragment_id: u64, physical_rows: u64, records: &[V0SeedRecord]) -> FragmentSeed {
        let schema = Arc::new(Schema::new(vec![
            Field::new("min", DataType::Float64, true),
            Field::new("max", DataType::Float64, true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("nan_count", DataType::UInt32, false),
            Field::new("zone_length", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(
                    records.iter().map(|r| r.0).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    records.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(UInt32Array::from_iter_values(records.iter().map(|r| r.2))),
                Arc::new(UInt32Array::from_iter_values(records.iter().map(|r| r.3))),
                Arc::new(UInt64Array::from_iter_values(records.iter().map(|r| r.4))),
            ],
        )
        .unwrap();
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = FileWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
            w.write(&batch).unwrap();
            w.finish().unwrap();
        }
        FragmentSeed {
            fragment_id,
            physical_rows,
            bytes: bytes::Bytes::from(buf.into_inner()),
            metadata_value: "0:2".to_string(),
        }
    }
    let mut v1 = train_and_load::<Float64Type>(vec![vec![Some(1.0), Some(2.0)]]).await;
    Arc::get_mut(&mut v1).unwrap().null_rows = None;
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    // null_count 2 in a record of length 1.
    let bad_counts = v0_seed(1, 1, &[(None, None, 2, 0, 1)]);
    assert!(
        v1.try_update_with_seeds(&[bad_counts], dest_store.as_ref())
            .await
            .unwrap()
            .is_none()
    );
    // Records covering 2 rows for a fragment of 3.
    let bad_coverage = v0_seed(1, 3, &[(Some(5.0), Some(6.0), 0, 0, 2)]);
    assert!(
        v1.try_update_with_seeds(&[bad_coverage], dest_store.as_ref())
            .await
            .unwrap()
            .is_none()
    );
    // The valid sibling is still admitted.
    let good = v0_seed(1, 2, &[(Some(5.0), Some(6.0), 0, 0, 2)]);
    let created = v1
        .try_update_with_seeds(&[good], dest_store.as_ref())
        .await
        .unwrap()
        .expect("a valid version 0 record is admitted");
    assert_eq!(created.index_version, 1);
}

/// The low-level update keeps the segment's file version: a version 0
/// segment stays version 0 (with the legacy marker for new NaN zones)
/// and a version 1 segment stays version 1.
#[tokio::test]
async fn test_update_keeps_file_version() {
    fn new_data() -> SendableRecordBatchStream {
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Float64, true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(vec![Some(7.0), Some(-f64::NAN)])),
                Arc::new(UInt64Array::from(vec![1u64 << 32, (1u64 << 32) | 1])),
            ],
        )
        .unwrap();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(batch))),
        ))
    }

    // Version 0 reference.
    let tmpdir = TempObjDir::default();
    let store = local_store(&tmpdir);
    write_v0_float_zonemap(store.as_ref(), &[(Some(1.0), Some(2.0), 0, 0, 0, 0, 2)]).await;
    let v0 = ZoneMapIndex::load(
        store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V0,
    )
    .await
    .unwrap();
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    let created = v0
        .update(new_data(), dest_store.as_ref(), None)
        .await
        .unwrap();
    assert_eq!(created.index_version, 0);
    let updated = ZoneMapIndex::load(
        dest_store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V0,
    )
    .await
    .unwrap();
    assert_eq!(updated.zones.len(), 2);
    assert_eq!(updated.zones[1].bound.fragment_id, 1);
    assert_eq!(updated.zones[1].nan_count, 1);
    // Written under the version 0 contract: marker max, split unknown.
    assert!(ZoneMapIndex::scalar_is_nan(&updated.zones[1].max));
    assert_eq!(updated.zones[1].negative_nan_count, None);
    assert!(
        ZoneMapIndex::load(
            dest_store,
            None,
            &LanceCache::no_cache(),
            false,
            ZoneMapIndexVersion::V1,
        )
        .await
        .is_err()
    );

    // Version 1 reference.
    let v1 = train_and_load::<Float64Type>(vec![vec![Some(1.0), Some(2.0)]]).await;
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    let created = v1
        .update(new_data(), dest_store.as_ref(), None)
        .await
        .unwrap();
    assert_eq!(created.index_version, 1);
    let updated = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert_eq!(updated.zones.len(), 2);
    assert_eq!(updated.zones[1].max, ScalarValue::Float64(Some(7.0)));
    assert_eq!(updated.zones[1].negative_nan_count, Some(1));
}

/// The low-level update adds the stream's zones beside the existing
/// ones: partial input for a fragment never erases that fragment's other
/// rows or null positions (whole-fragment supersession is the dataset
/// builder's, which knows the refreshed fragments).
#[tokio::test]
async fn test_update_is_additive_for_a_partially_fed_fragment() {
    use arrow::datatypes::Int32Type;

    let index = train_and_load::<Int32Type>(vec![vec![None, Some(2), Some(3), Some(4)]]).await;
    assert_eq!(index.zones.len(), 2);
    assert!(index.null_rows.as_ref().unwrap().contains(0));

    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Int32, true),
        Field::new(lance_core::ROW_ADDR, DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![Some(9), None, Some(11), Some(12)])),
            Arc::new(UInt64Array::from(vec![4u64, 5, 6, 7])),
        ],
    )
    .unwrap();
    let new_data: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::once(std::future::ready(Ok(batch))),
    ));
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    index
        .update(new_data, dest_store.as_ref(), None)
        .await
        .unwrap();
    let updated = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();

    let layout: Vec<(u64, u64, usize)> = updated
        .zones
        .iter()
        .map(|z| (z.bound.fragment_id, z.bound.start, z.bound.length))
        .collect();
    assert_eq!(layout, vec![(0, 0, 2), (0, 2, 2), (0, 4, 2), (0, 6, 2)]);
    let nulls = updated.null_rows.as_ref().unwrap();
    assert!(
        nulls.contains(0) && nulls.contains(5),
        "old and new null positions"
    );
    assert!(!nulls.contains(1) && !nulls.contains(4));
    for (value, row) in [(2, 1), (4, 3), (9, 4), (12, 7)] {
        let found = updated
            .search(
                &SargableQuery::Equals(ScalarValue::Int32(Some(value))),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        assert!(
            found.row_addrs().selected_rows().contains(row),
            "{value} at row {row}"
        );
    }
    let nulls_found = updated
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        nulls_found.row_addrs().selected_rows().contains(0)
            && nulls_found.row_addrs().selected_rows().contains(5)
    );
}

/// Dropping fragments from a bitmap removes their entries; an empty
/// entry would make an otherwise equal map compare unequal.
#[test]
fn test_without_fragments_drops_entries() {
    let mut rows = RowAddrTreeMap::new();
    rows.insert_range((1u64 << 32) | 3..(1u64 << 32) | 4);
    rows.insert_range((2u64 << 32) | 1..(2u64 << 32) | 2);
    let mut expected = RowAddrTreeMap::new();
    expected.insert_range((2u64 << 32) | 1..(2u64 << 32) | 2);
    let kept = super::without_fragments(&rows, [1]);
    assert_eq!(kept, expected);
    assert!(kept.get_fragment_bitmap(1).is_none());
    assert_eq!(
        super::without_fragments(&rows, [1, 2]),
        RowAddrTreeMap::new()
    );
    assert_eq!(super::without_fragments(&rows, [7]), rows);
}

/// Seed admission into a version 1 file: a version 0 reference is never
/// updated with seeds (the caller rebuilds), a version 0 seed record with
/// NaNs is not admitted (its split is unknown), and a version 0 seed
/// record without NaNs is admitted with a provably zero split.
#[tokio::test]
async fn test_update_with_seeds_version_admission() {
    use crate::scalar::seed::FragmentSeed;

    fn seed_bytes(values: &[Option<f64>]) -> bytes::Bytes {
        let mut writer = ZoneMapSeedWriter::new("c", 2, DataType::Float64).unwrap();
        let values: ArrayRef = Arc::new(Float64Array::from(values.to_vec()));
        writer.observe_batch(&values).unwrap();
        writer.finish().unwrap().unwrap()
    }
    fn seed(fragment_id: u64, bytes: bytes::Bytes) -> FragmentSeed {
        FragmentSeed {
            fragment_id,
            physical_rows: 2,
            bytes,
            metadata_value: "0:2".to_string(),
        }
    }

    let plain_v0 = to_v0(&seed_bytes(&[Some(5.0), Some(6.0)]));
    let nan_v0 = to_v0(&seed_bytes(&[Some(5.0), Some(f64::NAN)]));
    let nan_v1 = seed_bytes(&[Some(5.0), Some(-f64::NAN)]);

    // Version 0 reference: no seed update at all.
    let tmpdir = TempObjDir::default();
    let store = local_store(&tmpdir);
    write_v0_float_zonemap(store.as_ref(), &[(Some(1.0), Some(2.0), 0, 0, 0, 0, 2)]).await;
    let v0 = ZoneMapIndex::load(
        store,
        None,
        &LanceCache::no_cache(),
        true,
        ZoneMapIndexVersion::V0,
    )
    .await
    .unwrap();
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    assert!(
        v0.try_update_with_seeds(&[seed(1, plain_v0.clone())], dest_store.as_ref())
            .await
            .unwrap()
            .is_none()
    );

    // Version 1 reference. Its seeds carry no null positions (version 0
    // seeds never do), so the reference is one without a bitmap: with a
    // bitmap the positions would be required and the seeds scanned.
    let mut v1 = train_and_load::<Float64Type>(vec![vec![Some(1.0), Some(2.0)]]).await;
    Arc::get_mut(&mut v1).unwrap().null_rows = None;
    let dest_tmpdir = TempObjDir::default();
    let dest_store = local_store(&dest_tmpdir);
    assert!(
        v1.try_update_with_seeds(&[seed(1, nan_v0)], dest_store.as_ref())
            .await
            .unwrap()
            .is_none(),
        "a version 0 seed record with NaNs has an unknown split"
    );
    let created = v1
        .try_update_with_seeds(&[seed(1, plain_v0), seed(2, nan_v1)], dest_store.as_ref())
        .await
        .unwrap()
        .expect("version 1 and NaN-free version 0 seeds are admitted");
    assert_eq!(created.index_version, 1);
    let updated = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        true,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert_eq!(updated.zones.len(), 3);
    assert_eq!(updated.zones[1].bound.fragment_id, 1);
    assert_eq!(updated.zones[1].max, ScalarValue::Float64(Some(6.0)));
    assert_eq!(updated.zones[1].negative_nan_count, Some(0));
    assert_eq!(updated.zones[2].bound.fragment_id, 2);
    assert_eq!(updated.zones[2].max, ScalarValue::Float64(Some(5.0)));
    assert_eq!(updated.zones[2].nan_count, 1);
    assert_eq!(updated.zones[2].negative_nan_count, Some(1));
}

#[tokio::test]
async fn test_index_without_null_bitmap_keeps_that_setting() {
    let tmpdir = TempObjDir::default();
    let store = local_store(&tmpdir);
    let params = ZoneMapIndexBuilderParams::new(4).with_null_bitmap(false);
    let created = build_segment_from_batches(
        store.as_ref(),
        params,
        vec![vec![Some(1), None, Some(3), Some(4), None]],
    )
    .await;
    let details = created
        .index_details
        .to_msg::<pbold::ZoneMapIndexDetails>()
        .unwrap();
    assert_eq!(details.has_null_bitmap, Some(false));

    let index = ZoneMapIndex::load(
        store.clone(),
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert!(!index.store_null_bitmap());
    assert!(!index.has_null_bitmap());
    assert_eq!(index.num_zones(), 2);
    let bounds: Vec<(u64, usize)> = index
        .zones
        .iter()
        .map(|z| (z.bound.start, z.bound.length))
        .collect();
    assert_eq!(bounds, vec![(0, 4), (4, 1)]);
    let result = index
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(!result.is_exact());
    let derived = index.derive_index_params().unwrap();
    let derived: ZoneMapIndexBuilderParams =
        serde_json::from_str(derived.params.as_deref().unwrap()).unwrap();
    assert!(!derived.store_null_bitmap());

    // The legacy low-level update (a new fragment, after an empty batch
    // standing in for fragment 0) never regains a bitmap and keeps the
    // setting.
    let dest = TempObjDir::default();
    let dest_store = local_store(&dest);
    let schema = Arc::new(Schema::new(vec![Field::new(
        VALUE_COLUMN_NAME,
        DataType::Int32,
        true,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![None, Some(9)]))],
    )
    .unwrap();
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter([Ok(RecordBatch::new_empty(schema)), Ok(batch)]),
    ));
    let updated = index
        .update(add_row_addr(stream), dest_store.as_ref(), None)
        .await
        .unwrap();
    let details = updated
        .index_details
        .to_msg::<pbold::ZoneMapIndexDetails>()
        .unwrap();
    assert_eq!(details.has_null_bitmap, Some(false));
    let updated = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert!(!updated.store_null_bitmap());
    assert!(!updated.has_null_bitmap());
    assert_eq!(updated.num_zones(), 3);

    // A legacy index without the setting is still treated as wanting one.
    let legacy = train_and_load::<arrow::datatypes::Int32Type>(vec![vec![Some(1)]]).await;
    assert!(legacy.store_null_bitmap());
    assert!(legacy.has_null_bitmap());
}

#[tokio::test]
async fn test_merge_rejects_segments_with_different_null_bitmap_settings() {
    let with_dir = TempObjDir::default();
    let with_store = local_store(&with_dir);
    build_segment_from_batches(
        with_store.as_ref(),
        ZoneMapIndexBuilderParams::new(4),
        vec![vec![Some(1)]],
    )
    .await;
    let without_dir = TempObjDir::default();
    let without_store = local_store(&without_dir);
    build_segment_from_batches(
        without_store.as_ref(),
        ZoneMapIndexBuilderParams::new(4).with_null_bitmap(false),
        vec![vec![Some(2)]],
    )
    .await;
    let with = ZoneMapIndex::load(
        with_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    let without = ZoneMapIndex::load(
        without_store,
        None,
        &LanceCache::no_cache(),
        false,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    let dest = TempObjDir::default();
    let err = merge_zonemap_indices(
        &[&with, &without],
        local_store(&dest).as_ref(),
        &RoaringBitmap::from_iter([0u32, 1]),
    )
    .await
    .err()
    .expect("mixed settings must be rejected");
    assert!(err.to_string().contains("null bitmap settings"), "{err}");
}

/// An update whose scan covers a partial fragment's uncovered interval
/// and a whole fragment: zones are cut at the configured boundaries,
/// seed and scan fold into one zone, positions stay exact.
#[tokio::test]
async fn test_build_segment_update_folds_seeds_and_scan_into_target_zones() {
    let tmpdir = TempObjDir::default();
    let store = local_store(&tmpdir);
    // Fragment 0 from a scan, with a bitmap, target zones of 4.
    build_segment_from_batches(
        store.as_ref(),
        ZoneMapIndexBuilderParams::new(4),
        vec![vec![None, Some(1)]],
    )
    .await;
    let existing = ZoneMapIndex::load(
        store,
        None,
        &LanceCache::no_cache(),
        true,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    assert!(existing.has_null_bitmap());

    // Fragment 1: 6 rows in 3-row seed records [0,3) [3,6) with target 4:
    // record 0 fits zone 0, record 1 crosses -> rows [3,6) are scanned.
    // Values 0,3,null,9,null,15 (nulls at 2 and 4).
    let seed = seed_of(1, 3, int_values(6, 2));
    let mut reused = ReusedZones::new(true);
    let outcome = reused
        .reuse(&harvested(seed, 6), 4, &DataType::Int32, true)
        .unwrap();
    assert_eq!(outcome, SeedOutcome::Partial(vec![3..6]));
    // The scan supplies rows [3,6) of fragment 1 and all of fragment 2.
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Int32, true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let rows = |values: Vec<Option<i32>>, addrs: Vec<u64>| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(values)),
                Arc::new(UInt64Array::from(addrs)),
            ],
        )
        .unwrap()
    };
    let scanned: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter([
            Ok(rows(
                vec![Some(9), None, Some(15)],
                vec![(1 << 32) | 3, (1 << 32) | 4, (1 << 32) | 5],
            )),
            Ok(rows(vec![Some(5), None], vec![2 << 32, (2 << 32) | 1])),
        ]),
    ));
    let physical_rows = HashMap::from([(1u64, 6u64), (2, 2)]);
    let dest = TempObjDir::default();
    let dest_store = local_store(&dest);
    let created = build_zonemap_segment(
        ZoneMapSegmentBase::Update(&existing),
        &DataType::Int32,
        reused,
        Some(scanned),
        &physical_rows,
        &[1, 2],
        dest_store.as_ref(),
    )
    .await
    .unwrap();
    let details = created
        .index_details
        .to_msg::<pbold::ZoneMapIndexDetails>()
        .unwrap();
    assert_eq!(details.has_null_bitmap, Some(true));
    let updated = ZoneMapIndex::load(
        dest_store,
        None,
        &LanceCache::no_cache(),
        true,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    let zones: Vec<(u64, u64, usize, u32)> = updated
        .zones
        .iter()
        .map(|z| {
            (
                z.bound.fragment_id,
                z.bound.start,
                z.bound.length,
                z.null_count,
            )
        })
        .collect();
    // Fragment 1 zone 0 = seed record [0,3) (nulls at 0 and 2, value 3)
    // + scanned row 3 (value 9); zone 1 = scanned rows 4,5: null at 4.
    // Fragment 2: one zone of 2.
    assert_eq!(
        zones,
        vec![(0, 0, 2, 1), (1, 0, 4, 2), (1, 4, 2, 1), (2, 0, 2, 1)]
    );
    let zone = &updated.zones[1];
    assert_eq!(zone.min, ScalarValue::Int32(Some(3)));
    assert_eq!(zone.max, ScalarValue::Int32(Some(9)));
    let nulls = updated.null_rows.as_ref().unwrap();
    assert!(nulls.contains(0));
    assert!(nulls.contains(1 << 32));
    assert!(nulls.contains((1 << 32) | 2));
    assert!(nulls.contains((1 << 32) | 4));
    assert!(nulls.contains((2 << 32) | 1));
    assert_eq!(nulls.len(), Some(5));

    // Reusing seeds without positions for an index that keeps them is a
    // caller error, not a silent downgrade.
    let seed = seed_of(3, 4, int_values(2, 0));
    let mut no_positions = ReusedZones::new(false);
    no_positions
        .reuse(&harvested(seed, 2), 4, &DataType::Int32, true)
        .unwrap();
    let err = build_zonemap_segment(
        ZoneMapSegmentBase::Update(&existing),
        &DataType::Int32,
        no_positions,
        None,
        &HashMap::new(),
        &[3],
        local_store(&TempObjDir::default()).as_ref(),
    )
    .await
    .err()
    .expect("reuse without positions must be rejected");
    assert!(err.to_string().contains("without null positions"), "{err}");
}

/// The low-level plugin entry reuses only seeds whose every record fits
/// a target zone and falls back to a scan otherwise.
#[tokio::test]
async fn test_update_from_seeds_requires_whole_record_fit_and_positions() {
    let tmpdir = TempObjDir::default();
    let store = local_store(&tmpdir);
    build_segment_from_batches(
        store.as_ref(),
        ZoneMapIndexBuilderParams::new(4),
        vec![vec![None, Some(1)]],
    )
    .await;
    let existing = ZoneMapIndex::load(
        store,
        None,
        &LanceCache::no_cache(),
        true,
        ZoneMapIndexVersion::V1,
    )
    .await
    .unwrap();
    let dest = TempObjDir::default();
    let dest_store = local_store(&dest);

    // 4-row records into 4-row zones: reused.
    let fits = seed_of(1, 4, int_values(8, 3));
    let created = existing
        .try_update_with_seeds(&[fits], dest_store.as_ref())
        .await
        .unwrap();
    assert!(created.is_some());

    // 3-row records into 4-row zones cross a boundary: scan.
    let crosses = seed_of(2, 3, int_values(6, 0));
    assert!(
        existing
            .try_update_with_seeds(&[crosses], local_store(&TempObjDir::default()).as_ref())
            .await
            .unwrap()
            .is_none()
    );
    // A seed without positions for a bitmap-keeping index: scan.
    let old = without_null_positions(&seed_of(3, 4, int_values(4, 0)));
    assert!(
        existing
            .try_update_with_seeds(&[old], local_store(&TempObjDir::default()).as_ref())
            .await
            .unwrap()
            .is_none()
    );
}

/// Version 1 seed round trip: finite bounds, exact sign split, version
/// metadata; and the reader rejects unknown versions and invalid counts.
#[test]
fn test_seed_v1_round_trip_and_validation() {
    let mut writer = ZoneMapSeedWriter::new("c", 4, DataType::Float64).unwrap();
    let values: ArrayRef = Arc::new(Float64Array::from(vec![
        Some(1.0),
        Some(-f64::NAN),
        Some(f64::NAN),
        Some(5.0),
        Some(7.0),
        None,
    ]));
    writer.observe_batch(&values).unwrap();
    let bytes = writer.finish().unwrap().unwrap();

    let (zones, _bitmap) = ZoneMapSeedWriter::deserialize_seed(3, &bytes).unwrap();
    assert_eq!(zones.len(), 2);
    // Zone 0: [1.0, -NaN, +NaN, 5.0] -> finite bounds, exact split.
    assert_eq!(zones[0].min, ScalarValue::Float64(Some(1.0)));
    assert_eq!(zones[0].max, ScalarValue::Float64(Some(5.0)));
    assert_eq!(zones[0].nan_count, 2);
    assert_eq!(zones[0].negative_nan_count, Some(1));
    assert_eq!(zones[0].bound.fragment_id, 3);
    // Zone 1: [7.0, null].
    assert_eq!(zones[1].null_count, 1);
    assert_eq!(zones[1].negative_nan_count, Some(0));

    // Tampered version metadata is refused.
    let tampered = {
        use arrow_ipc::reader::FileReader;
        use arrow_ipc::writer::FileWriter;
        use std::io::Cursor;
        let mut reader = FileReader::try_new(Cursor::new(bytes.as_ref()), None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        let mut schema = batch.schema().as_ref().clone();
        schema
            .metadata
            .insert("version".to_string(), "2".to_string());
        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), batch.columns().to_vec()).unwrap();
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = FileWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
            w.write(&batch).unwrap();
            w.finish().unwrap();
        }
        bytes::Bytes::from(buf.into_inner())
    };
    let err = ZoneMapSeedWriter::deserialize_seed(0, &tampered).unwrap_err();
    assert!(err.to_string().contains("not supported"));
}

/// Version 0 seeds (no version metadata) keep their legacy semantics:
/// split unknown, and a legacy +NaN max marker is preserved as data.
#[test]
fn test_seed_v0_read_back() {
    use arrow_ipc::writer::FileWriter;
    use std::io::Cursor;
    // Hand-build a v0 seed batch: the old five-column schema, NaN max.
    let schema = Arc::new(Schema::new(vec![
        Field::new("min", DataType::Float64, true),
        Field::new("max", DataType::Float64, true),
        Field::new("null_count", DataType::UInt32, false),
        Field::new("nan_count", DataType::UInt32, false),
        Field::new("zone_length", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![Some(1.0)])),
            Arc::new(Float64Array::from(vec![Some(f64::NAN)])),
            Arc::new(UInt32Array::from(vec![0])),
            Arc::new(UInt32Array::from(vec![2])),
            Arc::new(UInt64Array::from(vec![4u64])),
        ],
    )
    .unwrap();
    let mut buf = Cursor::new(Vec::new());
    {
        let mut w = FileWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();
    }
    let bytes = bytes::Bytes::from(buf.into_inner());
    let (zones, bitmap) = ZoneMapSeedWriter::deserialize_seed(1, &bytes).unwrap();
    assert!(bitmap.is_none());
    assert_eq!(zones.len(), 1);
    assert_eq!(zones[0].negative_nan_count, None);
    assert!(ZoneMapIndex::scalar_is_nan(&zones[0].max));
    assert_eq!(zones[0].nan_count, 2);
}

/// A writer configured without positions writes counts only.
#[test]
fn test_seed_writer_without_positions_writes_counts_only() {
    use crate::scalar::seed::IndexSeedWriter;
    let values: ArrayRef = Arc::new(int_values(10, 3));
    let mut writer = ZoneMapSeedWriter::new("c", 4, DataType::Int32)
        .unwrap()
        .with_null_positions(false);
    writer.observe_batch(&values).unwrap();
    let bytes = writer.finish().unwrap().unwrap();
    let (records, positions) = ZoneMapSeedWriter::deserialize_seed(0, &bytes).unwrap();
    assert!(positions.is_none());
    let nulls: Vec<u32> = records.iter().map(|r| r.null_count).collect();
    assert_eq!(nulls, vec![2, 1, 1]);
}
