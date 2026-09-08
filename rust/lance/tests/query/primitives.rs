// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::datatypes::*;
use arrow_array::{
    ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, Int32Array,
    LargeBinaryArray, LargeStringArray, RecordBatch, RecordBatchIterator, StringArray,
    StringViewArray,
};
use arrow_schema::DataType;
use lance::Dataset;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::{InsertBuilder, WriteMode, WriteParams};

use lance::index::DatasetIndexExt;
use lance_datagen::{ArrayGeneratorExt, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;

use super::{assert_filter_ids, test_filter, test_scan, test_take};
use crate::utils::DatasetTestCases;

#[tokio::test]
async fn test_query_bool() {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col(
            "value",
            array::cycle_bool(vec![true, false]).with_random_nulls(0.1),
        )
        .into_batch_rows(RowCount::from(60))
        .unwrap();
    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            // TODO: fix bug with bitmap and btree https://github.com/lancedb/lance/issues/4756
            // TODO: fix bug with zone map https://github.com/lancedb/lance/issues/4758
            // TODO: Add boolean to bloom filter supported types https://github.com/lancedb/lance/issues/4757
            // [None, Some(IndexType::Bitmap), Some(IndexType::BTree), Some(IndexType::BloomFilter), Some(IndexType::ZoneMap)],
            [None],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value").await;
            test_filter(&original, &ds, "NOT value").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::int8(DataType::Int8)]
#[case::int16(DataType::Int16)]
#[case::int32(DataType::Int32)]
#[case::int64(DataType::Int64)]
#[case::uint8(DataType::UInt8)]
#[case::uint16(DataType::UInt16)]
#[case::uint32(DataType::UInt32)]
#[case::uint64(DataType::UInt64)]
async fn test_query_integer(#[case] data_type: DataType) {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();
    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::Bitmap),
                Some(IndexType::BTree),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value > 20").await;
            test_filter(&original, &ds, "NOT (value > 20)").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
            test_filter(&original, &ds, "(value != 0) OR (value < 20)").await;
            test_filter(&original, &ds, "NOT ((value != 0) OR (value < 20))").await;
            test_filter(
                &original,
                &ds,
                "(value != 5) OR ((value != 52) OR (value IS NULL))",
            )
            .await;
            test_filter(
                &original,
                &ds,
                "NOT ((value != 5) OR ((value != 52) OR (value IS NULL)))",
            )
            .await;
        })
        .await
}

/// Regression test: BTree OR on nullable column with value not in index.
///
/// When all non-null values are far from the equality value (e.g. all > 100,
/// query `!= 0`), the BTree's page lookup finds no pages containing that value.
/// Previously, null pages were not consulted for non-IsNull queries, so the
/// null set was empty and `NOT(x = 0)` would incorrectly pass all rows
/// (including NULLs). See also test_search_tracks_nulls_for_absent_value in
/// lance-index for a direct unit test of the BTree fix.
#[tokio::test]
async fn test_btree_nullable_or_with_absent_value() {
    // All non-null values are in [100..160], so value 0 never appears in the index.
    // ~33% of rows are NULL (every 3rd row).
    let value_array: Int32Array = (0..60)
        .map(|i| if i % 3 == 0 { None } else { Some(100 + i) })
        .collect();
    let id_array = Int32Array::from((0..60).collect::<Vec<i32>>());

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(id_array) as ArrayRef),
        ("value", Arc::new(value_array) as ArrayRef),
    ])
    .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types("value", [Some(IndexType::BTree)])
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_filter(&original, &ds, "(value != 0) OR (value < 5)").await;
            test_filter(&original, &ds, "NOT ((value != 0) OR (value < 5))").await;
            test_filter(&original, &ds, "value != 0").await;
            test_filter(&original, &ds, "NOT (value = 0)").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await;
}

#[tokio::test]
#[rstest::rstest]
#[case::float32(DataType::Float32)]
#[case::float64(DataType::Float64)]
async fn test_query_float(#[case] data_type: DataType) {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();
    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::BTree),
                Some(IndexType::Bitmap),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value > 0.5").await;
            test_filter(&original, &ds, "NOT (value > 0.5)").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
            test_filter(&original, &ds, "isnan(value)").await;
            test_filter(&original, &ds, "not isnan(value)").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
// Float16 is missing on purpose: `safe_coerce_scalar` has no Float16 arm, so
// `value < 0.0` against a Float16 column fails to resolve the literal long before
// any of this matters. See rust/lance-datafusion/src/expr.rs.
#[case::float32(DataType::Float32)]
#[case::float64(DataType::Float64)]
async fn test_query_float_special_values(#[case] data_type: DataType) {
    let value_array: Arc<dyn arrow_array::Array> = match data_type {
        DataType::Float32 => Arc::new(Float32Array::from(vec![
            Some(0.0_f32),
            Some(-0.0_f32),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
            Some(f32::NAN),
            Some(1.0_f32),
            Some(-1.0_f32),
            Some(f32::MIN),
            Some(f32::MAX),
            None,
        ])),
        DataType::Float64 => Arc::new(Float64Array::from(vec![
            Some(0.0_f64),
            Some(-0.0_f64),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(f64::NAN),
            Some(1.0_f64),
            Some(-1.0_f64),
            Some(f64::MIN),
            Some(f64::MAX),
            None,
        ])),
        _ => unreachable!(),
    };

    let id_array = Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>()));

    let batch =
        RecordBatch::try_from_iter(vec![("id", id_array as ArrayRef), ("value", value_array)])
            .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::BTree),
                Some(IndexType::Bitmap),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
            test_filter(&original, &ds, "isnan(value)").await;
            test_filter(&original, &ds, "not isnan(value)").await;

            // The remaining predicates compare against zero, where DataFusion
            // 54 answers by Arrow's total order: it ranks `-0.0` below `+0.0`
            // instead of treating the two encodings as one number the way
            // IEEE 754 and SQL do. That makes it useless as the reference, so
            // assert the rows. Ids are 0: +0.0, 1: -0.0, 2: +inf, 3: -inf,
            // 4: NaN, 5: 1.0, 6: -1.0, 7: MIN, 8: MAX, 9: NULL.
            for zero in ["0.0", "-0.0"] {
                assert_filter_ids(&ds, &format!("value < {zero}"), &[3, 6, 7]).await;
                assert_filter_ids(&ds, &format!("value <= {zero}"), &[0, 1, 3, 6, 7]).await;
                assert_filter_ids(&ds, &format!("value = {zero}"), &[0, 1]).await;
                assert_filter_ids(&ds, &format!("value != {zero}"), &[2, 3, 4, 5, 6, 7, 8]).await;
                // NaN is row 4. Arrow sorts it above every other value, so it
                // survives `>` and `>=`, which IEEE would reject. That gap is
                // not specific to zero and this rewrite leaves it alone.
                assert_filter_ids(&ds, &format!("value > {zero}"), &[2, 4, 5, 8]).await;
                assert_filter_ids(&ds, &format!("value >= {zero}"), &[0, 1, 2, 4, 5, 8]).await;
                // A literal on the left. DataFusion's canonicalizer swaps it back
                // before the rewrite runs, so this pins the answer rather than the
                // mirroring branch, which `a_literal_on_the_left_mirrors_the_operator`
                // owns and which SQL reaches only when the other side is not a
                // bare column.
                assert_filter_ids(&ds, &format!("{zero} > value"), &[3, 6, 7]).await;
                // BETWEEN only works because the simplifier expands it into two
                // comparisons before the rewrite runs.
                assert_filter_ids(&ds, &format!("value BETWEEN {zero} AND {zero}"), &[0, 1]).await;
                assert_filter_ids(
                    &ds,
                    &format!("value NOT BETWEEN {zero} AND {zero}"),
                    &[2, 3, 4, 5, 6, 7, 8],
                )
                .await;
                // An IN list gains the encoding it does not spell out.
                assert_filter_ids(&ds, &format!("value IN ({zero}, 1.0)"), &[0, 1, 5]).await;
                assert_filter_ids(
                    &ds,
                    &format!("value NOT IN ({zero}, 1.0)"),
                    &[2, 3, 4, 6, 7, 8],
                )
                .await;
                // Composed with NULL logic, where this index layer has broken before.
                assert_filter_ids(
                    &ds,
                    &format!("value != {zero} OR value IS NULL"),
                    &[2, 3, 4, 5, 6, 7, 8, 9],
                )
                .await;
            }
        })
        .await
}

/// A rewritten zero predicate still has to reach a scalar index. Without this,
/// the rewrite could reshape the predicate into something `maybe_indexed_column`
/// no longer recognizes, and every zero filter would quietly fall back to a full
/// scan plus refine while still returning the right rows. Only the default BTree
/// index is covered here; the other index types are exercised for row equality by
/// `test_query_float_special_values`, not for pushdown.
#[tokio::test]
async fn test_float_zero_predicate_uses_scalar_index() {
    let batch = RecordBatch::try_from_iter(vec![
        (
            "id",
            Arc::new(Int32Array::from_iter_values(0..4)) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Float64Array::from(vec![0.0, -0.0, 1.0, -1.0])) as ArrayRef,
        ),
    ])
    .unwrap();
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut ds = Dataset::write(reader, "memory://zero_index_pushdown", None)
        .await
        .unwrap();
    ds.create_index(
        &["value"],
        IndexType::Scalar,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();

    for predicate in ["value = 0.0", "value < 0.0", "value != 0.0"] {
        let plan = ds
            .scan()
            .filter(predicate)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            plan.contains("ScalarIndexQuery"),
            "`{predicate}` should use the scalar index, got plan:\n{plan}"
        );
        // The rewrite's output survives a second `optimize_expr`, which the scan
        // path does run, so the predicate must not appear twice.
        assert_eq!(
            plan.matches("value_idx").count(),
            1,
            "`{predicate}` should search the index once, got plan:\n{plan}"
        );
    }

    // Rows appended after the index is built are answered by the unindexed scan
    // while the rest come from the index. Both halves have to agree.
    let appended = RecordBatch::try_from_iter(vec![
        (
            "id",
            Arc::new(Int32Array::from_iter_values(4..8)) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Float64Array::from(vec![0.0, -0.0, 1.0, -1.0])) as ArrayRef,
        ),
    ])
    .unwrap();
    let ds = InsertBuilder::new(Arc::new(ds))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![appended])
        .await
        .unwrap();

    assert_filter_ids(&ds, "value = 0.0", &[0, 1, 4, 5]).await;
    assert_filter_ids(&ds, "value < 0.0", &[3, 7]).await;
    assert_filter_ids(&ds, "value >= 0.0", &[0, 1, 2, 4, 5, 6]).await;
}

#[tokio::test]
#[rstest::rstest]
#[case::date32(DataType::Date32)]
#[case::date64(DataType::Date64)]
async fn test_query_date(#[case] data_type: DataType) {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::Bitmap),
                Some(IndexType::BTree),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value < current_date()").await;
            // Mid-range literal: rand_type samples dates from the fixed range
            // [2023-01-01, 2024-01-01), so this splits the generated values
            test_filter(&original, &ds, "value > DATE '2023-07-01'").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::timestamp_second(DataType::Timestamp(TimeUnit::Second, None))]
#[case::timestamp_millisecond(DataType::Timestamp(TimeUnit::Millisecond, None))]
#[case::timestamp_microsecond(DataType::Timestamp(TimeUnit::Microsecond, None))]
#[case::timestamp_nanosecond(DataType::Timestamp(TimeUnit::Nanosecond, None))]
async fn test_query_timestamp(#[case] data_type: DataType) {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::BTree),
                Some(IndexType::Bitmap),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value < current_timestamp()").await;
            // Mid-range literal: rand_type samples timestamps from the fixed range
            // [2023-01-01, 2024-01-01), so this splits the generated values
            test_filter(&original, &ds, "value > TIMESTAMP '2023-07-01 00:00:00'").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::utf8(DataType::Utf8)]
#[case::large_utf8(DataType::LargeUtf8)]
// #[case::string_view(DataType::Utf8View)] // TODO: https://github.com/lancedb/lance/issues/5172
async fn test_query_string(#[case] data_type: DataType) {
    // Create arrays that include empty strings
    let string_values = vec![
        Some("hello"),
        Some("world"),
        Some(""),
        Some("test"),
        Some("data"),
        Some(""),
        None,
        Some("apple"),
        Some("zebra"),
        Some(""),
    ];

    let value_array: ArrayRef = match data_type {
        DataType::Utf8 => Arc::new(StringArray::from(string_values.clone())),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(string_values.clone())),
        DataType::Utf8View => Arc::new(StringViewArray::from(string_values.clone())),
        _ => unreachable!(),
    };

    let id_array = Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>()));

    let batch =
        RecordBatch::try_from_iter(vec![("id", id_array as ArrayRef), ("value", value_array)])
            .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::Bitmap),
                Some(IndexType::BTree),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value = 'hello'").await;
            test_filter(&original, &ds, "value != 'hello'").await;
            test_filter(&original, &ds, "value = ''").await;
            test_filter(&original, &ds, "value > 'hello'").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::binary(DataType::Binary)]
#[case::large_binary(DataType::LargeBinary)]
// #[case::binary_view(DataType::BinaryView)] // TODO: https://github.com/lancedb/lance/issues/5172
async fn test_query_binary(#[case] data_type: DataType) {
    // Create arrays that include empty binary
    let binary_values = vec![
        Some(b"hello".as_slice()),
        Some(b"world".as_slice()),
        Some(b"".as_slice()),
        Some(b"test".as_slice()),
        Some(b"data".as_slice()),
        Some(b"".as_slice()),
        None,
        Some(b"apple".as_slice()),
        Some(b"zebra".as_slice()),
        Some(b"".as_slice()),
    ];

    let value_array: ArrayRef = match data_type {
        DataType::Binary => Arc::new(BinaryArray::from(binary_values.clone())),
        DataType::LargeBinary => Arc::new(LargeBinaryArray::from(binary_values.clone())),
        DataType::BinaryView => Arc::new(BinaryViewArray::from(binary_values.clone())),
        _ => unreachable!(),
    };

    let id_array = Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>()));

    let batch =
        RecordBatch::try_from_iter(vec![("id", id_array as ArrayRef), ("value", value_array)])
            .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::Bitmap),
                Some(IndexType::BTree),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value = X'68656C6C6F'").await; // 'hello' in hex
            test_filter(&original, &ds, "value != X'68656C6C6F'").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
// TODO: Add Decimal32 and Decimal64 https://github.com/lancedb/lance/issues/5174
#[case::decimal128(DataType::Decimal128(38, 10))]
#[case::decimal256(DataType::Decimal256(76, 20))]
async fn test_query_decimal(#[case] data_type: DataType) {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            // NOTE: BloomFilter not supported for decimals
            [None, Some(IndexType::Bitmap), Some(IndexType::BTree)],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value > 0").await;
            test_filter(&original, &ds, "value < 0").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

/// Regression test: filtered scan panics after compaction with SRID when a
/// RangeWithBitmap segment appears after a Range segment in a fragment's
/// RowIdSequence. The bitmap iterator was advanced using a global offset
/// instead of a range-local position, exhausting the iterator.
///
/// Sequence: Write(2 frags) → Delete(from frag1) → Compact → CreateIndex → FilteredScan
#[tokio::test]
async fn test_filtered_scan_after_compact_with_srid() {
    use arrow::record_batch::RecordBatchIterator;

    // Write 100 rows across 2 fragments (50 each) with stable row IDs.
    let batch = RecordBatch::try_from_iter(vec![(
        "int_col",
        Arc::new(Int32Array::from_iter_values(0..100)) as ArrayRef,
    )])
    .unwrap();
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let write_params = WriteParams {
        enable_stable_row_ids: true,
        max_rows_per_file: 50,
        ..Default::default()
    };
    let mut ds = Dataset::write(reader, "memory://compact_srid_test", Some(write_params))
        .await
        .unwrap();
    assert_eq!(ds.get_fragments().len(), 2);
    assert_eq!(ds.count_rows(None).await.unwrap(), 100);

    // Delete some rows from the second fragment to create holes.
    // After compaction, this fragment's row_ids become a RangeWithBitmap segment.
    ds.delete("int_col >= 60 AND int_col < 70").await.unwrap();
    assert_eq!(ds.count_rows(None).await.unwrap(), 90);

    // Compact: merges both fragments into one. The output RowIdSequence has
    // multiple segments: Range(0..50) followed by RangeWithBitmap(50..100).
    // The RangeWithBitmap segment has offset_start=50 from the preceding Range.
    compact_files(&mut ds, CompactionOptions::default(), None)
        .await
        .unwrap();

    // Create a BTree index so filtered scans use mask_to_offset_ranges.
    ds.create_index(
        &["int_col"],
        IndexType::BTree,
        None,
        &lance_index::scalar::ScalarIndexParams::default(),
        true,
    )
    .await
    .unwrap();

    // Filtered scan: the index produces a RowAddrMask, which is passed to
    // mask_to_offset_ranges on the multi-segment RowIdSequence. Before the
    // fix, this panicked with "called Option::unwrap() on a None value".
    let results = ds
        .scan()
        .filter("int_col < 200")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(
        results.num_rows(),
        90,
        "Expected 90 rows (100 written - 10 deleted) but got {}",
        results.num_rows()
    );
}

/// Verifies that a zone map index on a string column is used (ScalarIndexQuery
/// in the plan) for both IS NULL and IS NOT NULL predicate filters.
///
/// IS NOT NULL must not silently fall back to a full scan when a zone map
/// index exists — both predicates should leverage the index.
#[tokio::test]
async fn test_zone_map_null_index_used() {
    // 6 non-null strings and 4 null values across 10 rows.
    let string_values = vec![
        Some("alpha"),
        None,
        Some("beta"),
        Some("gamma"),
        None,
        Some("delta"),
        None,
        Some("epsilon"),
        Some("zeta"),
        None,
    ];
    let value_array = Arc::new(StringArray::from(string_values)) as ArrayRef;
    let id_array = Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>())) as ArrayRef;
    let batch = RecordBatch::try_from_iter(vec![("id", id_array), ("value", value_array)]).unwrap();

    let mut ds = InsertBuilder::new("memory://")
        .execute(vec![batch])
        .await
        .unwrap();

    ds.create_index(
        &["value"],
        IndexType::ZoneMap,
        None,
        &lance_index::scalar::ScalarIndexParams::default(),
        true,
    )
    .await
    .unwrap();

    // IS NULL: the zone map index must appear in the plan.
    let plan = ds
        .scan()
        .filter("value IS NULL")
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(
        plan.contains("ScalarIndexQuery"),
        "IS NULL should use zone map index, got plan:\n{}",
        plan
    );
    let null_batch = ds
        .scan()
        .filter("value IS NULL")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(null_batch.num_rows(), 4);

    // IS NOT NULL: the zone map index must also appear in the plan.
    let plan = ds
        .scan()
        .filter("value IS NOT NULL")
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(
        plan.contains("ScalarIndexQuery"),
        "IS NOT NULL should use zone map index, got plan:\n{}",
        plan
    );
    let non_null_batch = ds
        .scan()
        .filter("value IS NOT NULL")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(non_null_batch.num_rows(), 6);
}
