// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! ZoneMap pruning with NaN of either sign.
//!
//! Arrow's total order puts a negative NaN below every value and a positive
//! NaN above, and the zone statistics do not record which signs a zone holds,
//! so a zone with NaNs must stay a candidate for range queries. These tests
//! write every NaN/null shape into its own zone and require that a query with
//! the index returns the rows a scan returns, before and after write seeds are
//! harvested into the index.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::{
    Array, Float16Array, Float32Array, Float64Array, Int64Array, RecordBatch, RecordBatchIterator,
    UInt32Array,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use bytes::Bytes;
use futures::TryStreamExt;
use lance_core::utils::tempfile::TempStrDir;
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::seed::SEED_META_KEY_PREFIX;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::ReadBatchParams;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;

use crate::Dataset;
use crate::dataset::WriteParams;
use crate::index::DatasetIndexExt;

const ROWS_PER_ZONE: usize = 8;
const FLOAT_COLUMNS: [&str; 3] = ["f16", "f32", "f64"];

/// One zone's worth of values, written identically into every float column.
#[derive(Clone, Copy)]
enum V {
    Num(f64),
    PosNan(u8),
    NegNan(u8),
    Null,
}

fn zones() -> Vec<[V; ROWS_PER_ZONE]> {
    use V::*;
    vec![
        [
            Num(-0.0),
            Num(0.0),
            Num(f64::NEG_INFINITY),
            Num(f64::INFINITY),
            Num(5.0),
            Num(8.0),
            Num(-3.0),
            Num(1.0),
        ],
        [
            NegNan(0),
            NegNan(1),
            Num(5.0),
            Num(8.0),
            Num(1.0),
            Num(-3.0),
            Num(0.0),
            Num(2.0),
        ],
        [
            PosNan(0),
            PosNan(1),
            Num(5.0),
            Num(8.0),
            Num(1.0),
            Num(-3.0),
            Num(0.0),
            Num(2.0),
        ],
        [
            NegNan(0),
            PosNan(0),
            Num(5.0),
            Num(8.0),
            Num(1.0),
            Num(-3.0),
            Null,
            Num(2.0),
        ],
        [
            PosNan(0),
            PosNan(1),
            PosNan(2),
            PosNan(3),
            PosNan(0),
            PosNan(1),
            PosNan(2),
            PosNan(3),
        ],
        [
            NegNan(0),
            NegNan(1),
            NegNan(2),
            NegNan(3),
            NegNan(0),
            NegNan(1),
            NegNan(2),
            NegNan(3),
        ],
        [Null, Null, Null, Null, Null, Null, Null, Null],
        [
            Null,
            NegNan(0),
            Null,
            NegNan(1),
            Null,
            NegNan(0),
            Null,
            NegNan(1),
        ],
        [
            Null,
            PosNan(0),
            Null,
            PosNan(1),
            Null,
            PosNan(0),
            Null,
            PosNan(1),
        ],
        [
            Null,
            Num(5.0),
            Null,
            Num(8.0),
            Null,
            Num(-3.0),
            Null,
            Num(1.0),
        ],
    ]
}

fn f64_of(v: V) -> Option<f64> {
    match v {
        V::Num(x) => Some(x),
        V::PosNan(p) => Some(f64::from_bits(0x7ff8_0000_0000_0000 | p as u64)),
        V::NegNan(p) => Some(f64::from_bits(0xfff8_0000_0000_0000 | p as u64)),
        V::Null => None,
    }
}

fn f32_of(v: V) -> Option<f32> {
    match v {
        V::Num(x) => Some(x as f32),
        V::PosNan(p) => Some(f32::from_bits(0x7fc0_0000 | p as u32)),
        V::NegNan(p) => Some(f32::from_bits(0xffc0_0000 | p as u32)),
        V::Null => None,
    }
}

fn f16_of(v: V) -> Option<half::f16> {
    match v {
        V::Num(x) => Some(half::f16::from_f64(x)),
        V::PosNan(p) => Some(half::f16::from_bits(0x7e00 | p as u16)),
        V::NegNan(p) => Some(half::f16::from_bits(0xfe00 | p as u16)),
        V::Null => None,
    }
}

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int64, false),
        ArrowField::new("f16", DataType::Float16, true),
        ArrowField::new("f32", DataType::Float32, true),
        ArrowField::new("f64", DataType::Float64, true),
    ]))
}

fn fragment_batch(first_id: i64) -> RecordBatch {
    let values: Vec<V> = zones().into_iter().flatten().collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(
                first_id..first_id + values.len() as i64,
            )),
            Arc::new(Float16Array::from_iter(values.iter().copied().map(f16_of))),
            Arc::new(Float32Array::from_iter(values.iter().copied().map(f32_of))),
            Arc::new(Float64Array::from_iter(values.iter().copied().map(f64_of))),
        ],
    )
    .unwrap()
}

fn zone_map_params() -> ScalarIndexParams {
    ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
        .with_params(&serde_json::json!({"rows_per_zone": ROWS_PER_ZONE, "use_seeds": true}))
}

/// One fragment, a seeded ZoneMap index per float column, then an appended
/// fragment whose data file carries a write seed for each column.
async fn write_dataset(uri: &str) -> Dataset {
    let rows = fragment_batch(0);
    let num_rows = rows.num_rows();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows)], schema()),
        uri,
        Some(WriteParams {
            max_rows_per_file: num_rows,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    for column in FLOAT_COLUMNS {
        dataset
            .create_index(
                &[column],
                IndexType::ZoneMap,
                None,
                &zone_map_params(),
                false,
            )
            .await
            .unwrap();
    }
    dataset
        .append(
            RecordBatchIterator::new([Ok(fragment_batch(num_rows as i64))], schema()),
            None,
        )
        .await
        .unwrap();
    dataset
}

async fn open_file(dataset: &Dataset, path: &Path) -> FileReader {
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler = scheduler
        .open_file(path, &CachedFileSize::unknown())
        .await
        .unwrap();
    FileReader::try_open(
        file_scheduler,
        None,
        Default::default(),
        &dataset.metadata_cache.file_metadata_cache(path),
        FileReaderOptions::default(),
    )
    .await
    .unwrap()
}

async fn read_all(reader: &FileReader) -> RecordBatch {
    let batches: Vec<RecordBatch> = reader
        .read_stream(
            ReadBatchParams::RangeFull,
            1024,
            4,
            FilterExpression::no_filter(),
        )
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    concat_batches(&batches[0].schema(), &batches).unwrap()
}

/// The ZoneMap index file of `column`: its zone batch and the file schema
/// metadata the index stores next to it.
async fn index_file(
    dataset: &Dataset,
    column: &str,
) -> (RecordBatch, std::collections::HashMap<String, String>) {
    let field_id = dataset.schema().field(column).unwrap().id;
    let indices = dataset.load_indices().await.unwrap();
    let index = indices
        .iter()
        .find(|index| index.fields.first() == Some(&field_id))
        .unwrap();
    let path = dataset
        .indices_dir()
        .join(index.uuid.to_string())
        .join("zonemap.lance");
    let reader = open_file(dataset, &path).await;
    let metadata = reader.metadata().file_schema.metadata.clone();
    (read_all(&reader).await, metadata)
}

/// The write seed for `column` in the data file of fragment `fragment`.
async fn seed(dataset: &Dataset, fragment: usize, column: &str) -> Bytes {
    let file = dataset.get_fragments()[fragment].metadata().files[0]
        .path
        .clone();
    let path = dataset.data_dir().join(file);
    let reader = open_file(dataset, &path).await;
    let value = reader
        .metadata()
        .file_schema
        .metadata
        .get(&format!("{SEED_META_KEY_PREFIX}{column}"))
        .unwrap()
        .clone();
    let buf_index: u32 = value.split(':').next().unwrap().parse().unwrap();
    reader.read_global_buffer(buf_index).await.unwrap()
}

fn seed_batch(bytes: &Bytes) -> RecordBatch {
    let mut reader =
        arrow_ipc::reader::FileReader::try_new(std::io::Cursor::new(bytes.as_ref()), None).unwrap();
    reader.next().unwrap().unwrap()
}

/// A NaN literal of the column's own type. A literal of another float type
/// would make the planner cast the column, which keeps the filter away from
/// the index.
fn nan_literal(column: &str) -> String {
    let data_type = match column {
        "f16" => "Float16",
        "f32" => "Float32",
        "f64" => "Float64",
        other => panic!("unexpected column {other}"),
    };
    format!("arrow_cast('NaN', '{data_type}')")
}

/// Filters covering every bound kind against ordinary values, nulls and
/// both NaN signs. Negating a NaN literal sets its sign bit.
fn predicates(column: &str) -> Vec<String> {
    let nan = nan_literal(column);
    [
        format!("{column} < 0"),
        format!("{column} <= 0"),
        format!("{column} > 100"),
        format!("{column} >= 100"),
        format!("{column} = 5"),
        format!("{column} = 100"),
        format!("{column} IN (5, 8)"),
        format!("{column} IN (100, -{nan})"),
        format!("{column} BETWEEN 0 AND 10"),
        format!("{column} BETWEEN -{nan} AND 0"),
        format!("{column} BETWEEN 100 AND {nan}"),
        format!("{column} = {nan}"),
        format!("{column} = -{nan}"),
        format!("{column} < {nan}"),
        format!("{column} > {nan}"),
        format!("{column} < -{nan}"),
        format!("{column} > -{nan}"),
        format!("{column} IS NULL"),
        format!("{column} IS NOT NULL"),
    ]
    .into_iter()
    .collect()
}

/// Whether the planner routes `filter` through the ZoneMap index.
async fn uses_zone_map(dataset: &Dataset, filter: &str) -> bool {
    let plan = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .filter(filter)
        .unwrap()
        .explain_plan(true)
        .await
        .unwrap();
    plan.contains("ScalarIndexQuery") && plan.contains("ZoneMap")
}

/// Representative predicates, with ordinary and NaN literals, must be planned
/// through the ZoneMap index for every column; otherwise comparing the index
/// path with a scan would compare a scan with itself.
async fn assert_representative_queries_use_index(dataset: &Dataset) {
    for column in FLOAT_COLUMNS {
        let nan = nan_literal(column);
        for filter in [
            format!("{column} < 0"),
            format!("{column} > 100"),
            format!("{column} = 5"),
            format!("{column} IN (5, 8)"),
            format!("{column} BETWEEN 0 AND 10"),
            format!("{column} < {nan}"),
            format!("{column} > -{nan}"),
            format!("{column} = {nan}"),
            format!("{column} = -{nan}"),
        ] {
            assert!(
                uses_zone_map(dataset, &filter).await,
                "{filter} bypassed the index"
            );
        }
    }
}

/// The typed NaN literals carry the intended sign: equality with the
/// canonical `+NaN` or `-NaN` hits exactly the rows written with that bit
/// pattern (per fragment, zones 2, 3, 4 and 8 hold six canonical positive
/// NaNs and zones 1, 3, 5 and 7 six canonical negative ones).
async fn assert_nan_literals_have_expected_sign(dataset: &Dataset, fragments: usize) {
    for column in FLOAT_COLUMNS {
        let nan = nan_literal(column);
        let positive = ids(dataset, &format!("{column} = {nan}"), true).await;
        let negative = ids(dataset, &format!("{column} = -{nan}"), true).await;
        assert_eq!(positive.len(), 6 * fragments, "{column} positive NaN rows");
        assert_eq!(negative.len(), 6 * fragments, "{column} negative NaN rows");
        assert!(positive.iter().all(|id| !negative.contains(id)));
    }
}

async fn ids(dataset: &Dataset, filter: &str, use_index: bool) -> Vec<i64> {
    let batch = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .filter(filter)
        .unwrap()
        .use_scalar_index(use_index)
        .try_into_batch()
        .await
        .unwrap();
    let mut ids: Vec<i64> = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    ids.sort_unstable();
    ids
}

/// Every predicate returns the same rows with the ZoneMap index as without it.
async fn assert_index_matches_scan(dataset: &Dataset) {
    for column in FLOAT_COLUMNS {
        for filter in predicates(column) {
            let with_index = ids(dataset, &filter, true).await;
            let scanned = ids(dataset, &filter, false).await;
            assert_eq!(with_index, scanned, "{filter}");
        }
    }
}

/// The `+NaN` max is a marker for a positive NaN only: a zone whose NaNs are
/// all negative keeps its ordinary maximum (null without ordinary values), and
/// `nan_count` is the only trace of its NaNs.
#[tokio::test]
async fn test_max_marks_positive_nan_only() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    for column in FLOAT_COLUMNS {
        let (batch, _) = index_file(&dataset, column).await;
        let max = batch.column_by_name("max").unwrap();
        let nan_counts = batch
            .column_by_name("nan_count")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        // Zone order follows `zones()` (per fragment of ten zones).
        let expected: [(bool, u32); 10] = [
            (false, 0), // ordinary only
            (false, 2), // negative NaN with values: ordinary max stays
            (true, 2),  // positive NaN with values
            (true, 2),  // both signs
            (true, 8),  // only positive NaN
            (false, 8), // only negative NaN: no ordinary max at all
            (false, 0), // only null
            (false, 4), // null and negative NaN
            (true, 4),  // null and positive NaN
            (false, 0), // values and null
        ];
        for (i, (nan_max, nan_count)) in expected.iter().cycle().take(batch.num_rows()).enumerate()
        {
            assert_eq!(nan_counts.value(i), *nan_count, "{column} zone {i}");
            assert_eq!(is_nan_at(max, i), *nan_max, "{column} zone {i} max {max:?}");
        }
        let seed = seed_batch(&seed(&dataset, 1, column).await);
        let seed_max = seed.column_by_name("max").unwrap();
        for (i, (nan_max, _)) in expected.iter().enumerate() {
            assert_eq!(is_nan_at(seed_max, i), *nan_max, "{column} seed zone {i}");
        }
        assert!(
            !is_nan_at(max, 1) && !max.is_null(1),
            "{column}: negative-only zone lost its max"
        );
        assert!(
            max.is_null(5),
            "{column}: all-negative-NaN zone should have no max"
        );
    }
}

fn is_nan_at(array: &arrow_array::ArrayRef, i: usize) -> bool {
    if array.is_null(i) {
        return false;
    }
    match array.data_type() {
        DataType::Float16 => array
            .as_any()
            .downcast_ref::<Float16Array>()
            .unwrap()
            .value(i)
            .is_nan(),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(i)
            .is_nan(),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(i)
            .is_nan(),
        other => panic!("unexpected {other:?}"),
    }
}

/// Every predicate returns the rows a scan returns, with the index built by a
/// scan and again after the appended fragment's seeds are harvested into it.
#[tokio::test]
async fn test_nan_queries_match_scan() {
    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    assert_representative_queries_use_index(&dataset).await;
    assert_nan_literals_have_expected_sign(&dataset, 2).await;
    assert_index_matches_scan(&dataset).await;

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    for column in FLOAT_COLUMNS {
        let (batch, _) = index_file(&dataset, column).await;
        assert_eq!(batch.num_rows(), 2 * zones().len(), "{column}");
    }
    assert_representative_queries_use_index(&dataset).await;
    assert_index_matches_scan(&dataset).await;
}
