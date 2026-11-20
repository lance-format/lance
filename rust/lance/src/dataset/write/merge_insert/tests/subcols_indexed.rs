// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// Four rows keyed by `interval_idx` with `cells` (a `list<struct>`) null,
/// and a BTree index on the key so the merge takes the indexed join path.
async fn indexed_list_struct_dataset(target_keys: &[i64]) -> Dataset {
    let cells_type = DataType::List(Arc::new(Field::new(
        "item",
        DataType::Struct(
            vec![
                Field::new("log_time", DataType::Int64, true),
                Field::new(
                    "header",
                    DataType::Struct(
                        vec![
                            Field::new("seq", DataType::UInt32, true),
                            Field::new("frame_id", DataType::Utf8, true),
                        ]
                        .into(),
                    ),
                    true,
                ),
                Field::new("data", DataType::LargeBinary, true),
            ]
            .into(),
        ),
        true,
    )));
    let schema = Arc::new(Schema::new(vec![
        Field::new("recording_id", DataType::Utf8, true),
        Field::new("interval_idx", DataType::Int64, true),
        Field::new("t_start", DataType::Int64, true),
        Field::new("duration_ns", DataType::Int64, true),
        Field::new("cells", cells_type.clone(), true),
    ]));
    let num_rows = target_keys.len();
    let target = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["rec"; num_rows])),
            Arc::new(Int64Array::from(target_keys.to_vec())),
            Arc::new(Int64Array::from(vec![0i64; num_rows])),
            Arc::new(Int64Array::from(vec![1i64; num_rows])),
            arrow_array::new_null_array(&cells_type, num_rows),
        ],
    )
    .unwrap();
    let mut ds = Dataset::write(
        RecordBatchIterator::new([Ok(target)], schema),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    ds.create_index(
        &["interval_idx"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    ds
}
/// Two `cells` values, one cell each (`data` = "a" and "bb"). With
/// `reverse_nested` the struct children are declared in the reverse of
/// the dataset's order.
fn list_struct_cells(reverse_nested: bool) -> ListArray {
    use arrow_array::{LargeBinaryArray, UInt32Array};
    let mut header_fields = vec![
        Field::new("seq", DataType::UInt32, true),
        Field::new("frame_id", DataType::Utf8, true),
    ];
    let mut header_arrays: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from(vec![7, 8])),
        Arc::new(StringArray::from(vec!["base", "map"])),
    ];
    if reverse_nested {
        header_fields.reverse();
        header_arrays.reverse();
    }
    let header = StructArray::new(header_fields.into(), header_arrays, None);
    let mut cell_fields = vec![
        Field::new("log_time", DataType::Int64, true),
        Field::new("header", header.data_type().clone(), true),
        Field::new("data", DataType::LargeBinary, true),
    ];
    let mut cell_arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![100, 200])),
        Arc::new(header),
        Arc::new(LargeBinaryArray::from(vec![&b"a"[..], &b"bb"[..]])),
    ];
    if reverse_nested {
        cell_fields.reverse();
        cell_arrays.reverse();
    }
    let cells = StructArray::new(cell_fields.into(), cell_arrays, None);
    ListArray::new(
        Arc::new(Field::new("item", cells.data_type().clone(), true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 1, 2])),
        Arc::new(cells),
        None,
    )
}
async fn merge_indexed(
    ds: Dataset,
    source: RecordBatch,
    when_matched: WhenMatched,
    use_index: bool,
) -> Result<(Arc<Dataset>, MergeStats)> {
    let schema = source.schema();
    MergeInsertBuilder::try_new(Arc::new(ds), vec!["interval_idx".into()])?
        .when_matched(when_matched)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .use_index(use_index)
        .try_build()?
        .execute_reader(RecordBatchIterator::new([Ok(source)], schema))
        .await
}
/// Keys 0 and 1 must hold the cells "a" and "bb"; every other key stays null.
async fn assert_cells_filled(ds: &Dataset) {
    let after = ds
        .scan()
        .project(&["interval_idx", "cells"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let keys = after["interval_idx"].as_primitive::<arrow_array::types::Int64Type>();
    let cells = after["cells"].as_list::<i32>();
    for row in 0..after.num_rows() {
        let key = keys.value(row);
        assert_eq!(cells.is_valid(row), key < 2, "row {row} key {key}");
        if key < 2 {
            let cell = cells.value(row);
            let data = cell.as_struct()["data"]
                .as_binary::<i64>()
                .value(0)
                .to_vec();
            assert_eq!(
                data,
                if key == 0 {
                    b"a".to_vec()
                } else {
                    b"bb".to_vec()
                }
            );
        }
    }
}
/// A partial-schema source joined through the scalar index carries
/// `_rowaddr` next to `_rowid`; splitting the matched batch at its midpoint
/// built the `update_if` target struct from the wrong columns.
#[rstest::rstest]
#[case::unique_keys(vec![0, 1, 2, 3], true)]
#[case::unique_keys_no_condition(vec![0, 1, 2, 3], false)]
#[case::duplicate_keys(vec![0, 0, 1, 1], true)]
#[case::duplicate_keys_no_condition(vec![0, 0, 1, 1], false)]
#[tokio::test]
async fn test_indexed_partial_update_if_list_struct(
    #[case] target_keys: Vec<i64>,
    #[case] with_condition: bool,
) {
    let ds = indexed_list_struct_dataset(&target_keys).await;
    let cells = list_struct_cells(false);
    let schema = Arc::new(Schema::new(vec![
        Field::new("interval_idx", DataType::Int64, true),
        Field::new("cells", cells.data_type().clone(), true),
    ]));
    let source = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![0i64, 1])), Arc::new(cells)],
    )
    .unwrap();
    let when_matched = if with_condition {
        WhenMatched::update_if(&ds, "target.cells IS NULL").unwrap()
    } else {
        WhenMatched::UpdateAll
    };
    let (ds, stats) = merge_indexed(ds, source, when_matched, true).await.unwrap();
    let expected_updates = target_keys.iter().filter(|k| **k < 2).count();
    assert_eq!(stats.num_updated_rows, expected_updates as u64);
    assert_cells_filled(&ds).await;
}
/// The `update_if` target struct is assembled by name with the dataset's
/// field types: the joined target columns come back in dataset order, and
/// a source may declare a column non-nullable or order a nested struct's
/// children differently from the dataset.
///
/// The non-nullable case rewrites every target row: the column patch path
/// cannot yet fill a non-nullable source column with a fragment's nulls.
#[rstest::rstest]
#[case::column_order(false, true, vec![0, 1, 2, 3])]
#[case::non_nullable(false, false, vec![0, 0, 1, 1])]
#[case::nested_field_order(true, true, vec![0, 1, 2, 3])]
#[tokio::test]
async fn test_indexed_partial_update_if_source_shape(
    #[case] reverse_nested: bool,
    #[case] nullable: bool,
    #[case] target_keys: Vec<i64>,
    #[values(false, true)] use_index: bool,
) {
    let ds = indexed_list_struct_dataset(&target_keys).await;
    let cells = list_struct_cells(reverse_nested);
    let schema = Arc::new(Schema::new(vec![
        Field::new("cells", cells.data_type().clone(), nullable),
        Field::new("interval_idx", DataType::Int64, true),
    ]));
    let source = RecordBatch::try_new(
        schema,
        vec![Arc::new(cells), Arc::new(Int64Array::from(vec![0i64, 1]))],
    )
    .unwrap();
    let condition = WhenMatched::update_if(&ds, "target.cells IS NULL").unwrap();
    let (ds, stats) = merge_indexed(ds, source, condition, use_index)
        .await
        .unwrap();
    let expected_updates = target_keys.iter().filter(|k| **k < 2).count();
    assert_eq!(stats.num_updated_rows, expected_updates as u64);
    assert_cells_filled(&ds).await;
}
/// With same-typed columns a positional split would silently read the
/// wrong target column: here `target.duration_ns` must not read
/// `interval_idx`, which sits first in dataset order.
#[tokio::test]
async fn test_indexed_partial_update_if_same_type_column_order() {
    let ds = indexed_list_struct_dataset(&[0, 1, 2, 3]).await;
    let source = record_batch!(
        ("duration_ns", Int64, [5, 5]),
        ("t_start", Int64, [9, 9]),
        ("interval_idx", Int64, [0, 1])
    )
    .unwrap();
    let condition = WhenMatched::update_if(&ds, "target.duration_ns = 1").unwrap();
    let (ds, stats) = merge_indexed(ds, source, condition, true).await.unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    let after = ds
        .scan()
        .project(&["interval_idx", "duration_ns"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let keys = after["interval_idx"].as_primitive::<arrow_array::types::Int64Type>();
    let durations = after["duration_ns"].as_primitive::<arrow_array::types::Int64Type>();
    for row in 0..after.num_rows() {
        let expected = if keys.value(row) < 2 { 5 } else { 1 };
        assert_eq!(durations.value(row), expected, "row {row}");
    }
}
/// The indexed take projects `cells` to the children the source supplies, so
/// the joined `target_cells` is `list<struct<idx, data>>` while the dataset
/// holds `list<struct<idx, timestamp, data>>`. The condition must be planned
/// against the type the join emits.
#[rstest::rstest]
#[case::update_all(false)]
#[case::with_condition(true)]
#[tokio::test]
async fn test_indexed_partial_update_if_nested_subset(#[case] with_condition: bool) {
    use arrow_array::{Int32Array, LargeBinaryArray};
    let list_of = |fields: Vec<Field>| {
        DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(fields.into()),
            true,
        )))
    };
    let full = list_of(vec![
        Field::new("idx", DataType::Int32, true),
        Field::new("timestamp", DataType::Int64, true),
        Field::new("data", DataType::LargeBinary, true),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("interval_idx", DataType::Int64, true),
        Field::new("t_start", DataType::Int64, true),
        Field::new("cells", full.clone(), true),
    ]));
    let target = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![0i64, 1, 2, 3])),
            Arc::new(Int64Array::from(vec![0i64; 4])),
            arrow_array::new_null_array(&full, 4),
        ],
    )
    .unwrap();
    let mut ds = Dataset::write(
        RecordBatchIterator::new([Ok(target)], schema),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    ds.create_index(
        &["interval_idx"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();

    let subset = StructArray::new(
        vec![
            Field::new("idx", DataType::Int32, true),
            Field::new("data", DataType::LargeBinary, true),
        ]
        .into(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(LargeBinaryArray::from(vec![&b"a"[..], &b"bb"[..]])),
        ],
        None,
    );
    let cells = ListArray::new(
        Arc::new(Field::new("item", subset.data_type().clone(), true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 1, 2])),
        Arc::new(subset),
        None,
    );
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("interval_idx", DataType::Int64, true),
        Field::new("cells", cells.data_type().clone(), true),
    ]));
    let source = RecordBatch::try_new(
        source_schema,
        vec![Arc::new(Int64Array::from(vec![0i64, 1])), Arc::new(cells)],
    )
    .unwrap();
    let when_matched = if with_condition {
        WhenMatched::update_if(&ds, "target.cells IS NULL").unwrap()
    } else {
        WhenMatched::UpdateAll
    };
    let (ds, stats) = merge_indexed(ds, source, when_matched, true).await.unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    assert_cells_filled(&ds).await;
}

/// Six rows `id` 0..6 with `a = id * 10` and `b = "b{id}"` (or a struct `b` of
/// `x = id`, `y = "b{id}"`), two rows per fragment, optionally with `id = 0`
/// deleted, and a BTree index on `id`.
async fn indexed_condition_dataset(
    enable_stable_row_ids: bool,
    with_deletions: bool,
    struct_column: bool,
) -> Dataset {
    let ids = Int64Array::from_iter_values(0..6);
    let a = Int64Array::from_iter_values((0..6).map(|id| id * 10));
    let b = StringArray::from_iter_values((0..6).map(|id| format!("b{id}")));
    let (b_field, b): (Field, ArrayRef) = if struct_column {
        let s = StructArray::new(
            vec![
                Field::new("x", DataType::Int64, true),
                Field::new("y", DataType::Utf8, true),
            ]
            .into(),
            vec![Arc::new(ids.clone()), Arc::new(b)],
            None,
        );
        (Field::new("b", s.data_type().clone(), true), Arc::new(s))
    } else {
        (Field::new("b", DataType::Utf8, true), Arc::new(b))
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("a", DataType::Int64, true),
        b_field,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(a), b]).unwrap();
    let mut ds = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    if with_deletions {
        ds.delete("id = 0").await.unwrap();
    }
    ds.create_index(
        &["id"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    ds
}

/// `(id, a, b)` for every row, sorted by `id`; a struct `b` reports its `y`.
async fn id_a_b_rows(ds: &Dataset) -> Vec<(i64, i64, String)> {
    let batch = ds.scan().try_into_batch().await.unwrap();
    let ids = batch["id"].as_primitive::<arrow_array::types::Int64Type>();
    let a = batch["a"].as_primitive::<arrow_array::types::Int64Type>();
    let b = match batch["b"].data_type() {
        DataType::Struct(_) => batch["b"].as_struct()["y"].clone(),
        _ => batch["b"].clone(),
    };
    let b = b.as_string::<i32>();
    let mut rows = (0..batch.num_rows())
        .map(|i| (ids.value(i), a.value(i), b.value(i).to_string()))
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

/// The rows of [`indexed_condition_dataset`] from `first_id` through `last_id`,
/// with `b` replaced by `B{id}` for each id in `updated`.
fn expected_rows(first_id: i64, last_id: i64, updated: &[i64]) -> Vec<(i64, i64, String)> {
    (first_id..=last_id)
        .map(|id| {
            let b = if updated.contains(&id) { "B" } else { "b" };
            (id, id * 10, format!("{b}{id}"))
        })
        .collect()
}

fn when_matched_a_over_25(ds: &Dataset, condition: &str) -> WhenMatched {
    match condition {
        "expr" => {
            WhenMatched::update_if_expr(logical_expr::col("target.a").gt(logical_expr::lit(25i64)))
        }
        condition => WhenMatched::update_if(ds, condition).unwrap(),
    }
}

/// Merge `source` keyed on `id`, updating the matched rows `condition` accepts.
async fn merge_on_id(
    ds: &Dataset,
    source: RecordBatch,
    condition: &str,
    insert: bool,
    write_mode: MergeInsertWriteMode,
    use_index: bool,
) -> Result<(Arc<Dataset>, MergeStats)> {
    let schema = source.schema();
    MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".into()])?
        .when_matched(when_matched_a_over_25(ds, condition))
        .when_not_matched(if insert {
            WhenNotMatched::InsertAll
        } else {
            WhenNotMatched::DoNothing
        })
        .write_mode(write_mode)
        .use_index(use_index)
        .try_build()?
        .execute_reader(RecordBatchIterator::new([Ok(source)], schema))
        .await
}

/// A partial source `(b, id)` with `b = "B{id}"`.
fn partial_source(ids: Vec<i64>) -> RecordBatch {
    let b = StringArray::from_iter_values(ids.iter().map(|id| format!("B{id}")));
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("b", DataType::Utf8, true),
            Field::new("id", DataType::Int64, true),
        ])),
        vec![Arc::new(b), Arc::new(Int64Array::from(ids))],
    )
    .unwrap()
}

/// An `update_if` condition may read a target column the partial-schema source
/// omits. Creating a scalar index on the join key must not change that: the
/// indexed join only took the source's columns from the target, so the
/// condition failed to plan ("Field a not found in struct"). `RewriteRows` is
/// left out: with a partial source it gives up the index for the v2 plan.
#[rstest::rstest]
#[tokio::test]
async fn test_indexed_partial_update_if_reads_omitted_column(
    #[values("target.a > 25", "a > 25", "target.a > 25 AND source.b <> 'x'", "expr")]
    condition: &str,
    #[values(MergeInsertWriteMode::Auto, MergeInsertWriteMode::RewriteColumns)]
    write_mode: MergeInsertWriteMode,
    #[values((false, false), (true, true))] stable_row_ids_and_deletions: (bool, bool),
) {
    let (enable_stable_row_ids, with_deletions) = stable_row_ids_and_deletions;
    let ds = indexed_condition_dataset(enable_stable_row_ids, with_deletions, false).await;
    let source = partial_source(vec![1, 2, 3, 4, 9]);
    let (ds, stats) = merge_on_id(&ds, source, condition, false, write_mode, true)
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    let first = i64::from(with_deletions);
    assert_eq!(id_a_b_rows(&ds).await, expected_rows(first, 5, &[3, 4]));
}

/// The unindexed (v2) plan gives the same result for the same conditions.
#[rstest::rstest]
#[tokio::test]
async fn test_unindexed_partial_update_if_reads_omitted_column(
    #[values("target.a > 25", "a > 25", "target.a > 25 AND source.b <> 'x'", "expr")]
    condition: &str,
) {
    let ds = indexed_condition_dataset(false, false, false).await;
    let source = partial_source(vec![1, 2, 3, 4, 9]);
    let (ds, stats) = merge_on_id(
        &ds,
        source,
        condition,
        false,
        MergeInsertWriteMode::Auto,
        false,
    )
    .await
    .unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    assert_eq!(id_a_b_rows(&ds).await, expected_rows(0, 5, &[3, 4]));
}

/// Rows in a fragment written after the index are read by a fallback scan
/// that is unioned with the indexed take, so that scan must carry the
/// condition's column as well.
#[rstest::rstest]
#[tokio::test]
async fn test_indexed_partial_update_if_reads_omitted_column_in_unindexed_fragment(
    #[values("target.a > 25", "expr")] condition: &str,
    #[values(MergeInsertWriteMode::Auto, MergeInsertWriteMode::RewriteColumns)]
    write_mode: MergeInsertWriteMode,
) {
    let mut ds = indexed_condition_dataset(false, false, false).await;
    let appended = record_batch!(
        ("id", Int64, [6, 7]),
        ("a", Int64, [60, 70]),
        ("b", Utf8, ["b6", "b7"])
    )
    .unwrap();
    let schema = appended.schema();
    ds.append(RecordBatchIterator::new([Ok(appended)], schema), None)
        .await
        .unwrap();
    let source = partial_source(vec![2, 3, 7]);
    let (ds, stats) = merge_on_id(&ds, source, condition, false, write_mode, true)
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    assert_eq!(id_a_b_rows(&ds).await, expected_rows(0, 7, &[3, 7]));
}

/// The indexed join of a full-schema source also runs through the merger: a
/// `target.`-qualified DataFusion expression must resolve there as it does in
/// the v2 plan.
#[rstest::rstest]
#[tokio::test]
async fn test_indexed_full_schema_update_if(
    #[values("target.a > 25", "target.a > 25 AND source.id < 9", "expr")] condition: &str,
    #[values(false, true)] insert: bool,
) {
    let ds = indexed_condition_dataset(false, true, false).await;
    let source = record_batch!(
        ("id", Int64, [1, 2, 3, 4, 9]),
        ("a", Int64, [10, 20, 30, 40, 90]),
        ("b", Utf8, ["B1", "B2", "B3", "B4", "B9"])
    )
    .unwrap();
    let (ds, stats) = merge_on_id(
        &ds,
        source,
        condition,
        insert,
        MergeInsertWriteMode::Auto,
        true,
    )
    .await
    .unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    let mut expected = expected_rows(1, 5, &[3, 4]);
    if insert {
        expected.push((9, 90, "B9".into()));
    }
    assert_eq!(id_a_b_rows(&ds).await, expected);
}

/// A source that omits a nested child (`b.x`) is neither the full schema nor
/// a top-level subset, so the merger handles it with or without the index.
/// The condition reads the top-level column `a`, which the source also omits.
#[rstest::rstest]
#[tokio::test]
async fn test_nested_subset_update_if_reads_omitted_column(
    #[values("target.a > 25", "target.a > 25 AND source.id < 9", "expr")] condition: &str,
    #[values(false, true)] use_index: bool,
) {
    let ds = indexed_condition_dataset(false, true, true).await;
    let b = StructArray::new(
        vec![Field::new("y", DataType::Utf8, true)].into(),
        vec![Arc::new(StringArray::from(vec![
            "B1", "B2", "B3", "B4", "B9",
        ]))],
        None,
    );
    let source = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("b", b.data_type().clone(), true),
        ])),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4, 9])), Arc::new(b)],
    )
    .unwrap();
    let (ds, stats) = merge_on_id(
        &ds,
        source,
        condition,
        false,
        MergeInsertWriteMode::Auto,
        use_index,
    )
    .await
    .unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    assert_eq!(id_a_b_rows(&ds).await, expected_rows(1, 5, &[3, 4]));
}

/// An unqualified column both sides carry is ambiguous, with or without the
/// scalar index on the join key.
#[rstest::rstest]
#[tokio::test]
async fn test_update_if_unqualified_shared_column_is_ambiguous(
    #[values(false, true)] partial: bool,
    #[values(false, true)] use_index: bool,
) {
    let ds = indexed_condition_dataset(false, false, false).await;
    let source = if partial {
        partial_source(vec![3])
    } else {
        record_batch!(("id", Int64, [3]), ("a", Int64, [30]), ("b", Utf8, ["B3"])).unwrap()
    };
    let err = merge_on_id(
        &ds,
        source,
        "b <> 'x'",
        false,
        MergeInsertWriteMode::Auto,
        use_index,
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("Ambiguous reference"), "{err}");
}

/// `list<struct<v>>` values with one element each.
fn single_tag_lists(values: &[&str]) -> ArrayRef {
    let mut builder = ListBuilder::new(arrow_array::builder::StructBuilder::from_fields(
        vec![Field::new("v", DataType::Utf8, true)],
        values.len(),
    ));
    for value in values {
        builder
            .values()
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value(*value);
        builder.values().append(true);
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn struct_of_x(xs: Vec<i64>) -> ArrayRef {
    Arc::new(StructArray::new(
        vec![Field::new("x", DataType::Int64, true)].into(),
        vec![Arc::new(Int64Array::from(xs))],
        None,
    ))
}

/// Conditions over nested fields, inside function calls and in lambdas keep
/// resolving on the indexed join, as they did before: `s` and `tags` are in
/// the source, `a` is not. Rows 3 and 4 satisfy each condition; rows 1 and 2
/// do not.
#[rstest::rstest]
#[tokio::test]
async fn test_indexed_update_if_nested_and_function_forms(
    #[values(
        "target.s.x > 2",
        "coalesce(target.s.x, 0) > 2",
        "target.s.x > 2 AND source.s.x > 0",
        "array_any_match(target.tags, t -> t.v = 'hit')",
        "coalesce(target.a, 0) > 25"
    )]
    condition: &str,
) {
    let target_tags = single_tag_lists(&["miss", "miss", "miss", "hit", "hit", "hit"]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("a", DataType::Int64, true),
        Field::new("s", struct_of_x(vec![]).data_type().clone(), true),
        Field::new("tags", target_tags.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..6)),
            Arc::new(Int64Array::from_iter_values((0..6).map(|id| id * 10))),
            struct_of_x((0..6).collect()),
            target_tags,
        ],
    )
    .unwrap();
    let mut ds = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema.clone()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    ds.create_index(
        &["id"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    let source_schema = Arc::new(schema.project(&[0, 2, 3]).unwrap());
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            struct_of_x(vec![101, 102, 103, 104]),
            single_tag_lists(&["new", "new", "new", "new"]),
        ],
    )
    .unwrap();
    let (ds, stats) = MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".into()])
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, condition).unwrap())
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new([Ok(source)], source_schema))
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    let after = ds
        .scan()
        .project(&["id", "s"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = after["id"].as_primitive::<arrow_array::types::Int64Type>();
    let xs = after["s"].as_struct()["x"].clone();
    let xs = xs.as_primitive::<arrow_array::types::Int64Type>();
    let mut rows = (0..after.num_rows())
        .map(|i| (ids.value(i), xs.value(i)))
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        vec![(0, 0), (1, 1), (2, 2), (3, 103), (4, 104), (5, 5)]
    );
}

/// A literal compared with a `Float32` column the source omits is narrowed to
/// the column's type, as for any filter: `score = 0.1` matches the stored
/// `0.1f32`. An unqualified column only becomes a target field once bound, so
/// the literal must be resolved against the bound condition. The name is also
/// matched ignoring case, as the planner does.
#[rstest::rstest]
#[tokio::test]
async fn test_update_if_float32_omitted_column_literal(
    #[values("target.score = 0.1", "score = 0.1", "SCORE = 0.1")] condition: &str,
    #[values(false, true)] use_index: bool,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("score", DataType::Float32, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..6)),
            Arc::new(Float32Array::from_iter_values(
                (0..6).map(|id| id as f32 / 10.0),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..6).map(|id| format!("b{id}")),
            )),
        ],
    )
    .unwrap();
    let mut ds = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    ds.create_index(
        &["id"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    let source = partial_source(vec![1, 2]);
    let schema = source.schema();
    let (_, stats) = MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".into()])
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, condition).unwrap())
        .when_not_matched(WhenNotMatched::DoNothing)
        .use_index(use_index)
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new([Ok(source)], schema))
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 1);
}
