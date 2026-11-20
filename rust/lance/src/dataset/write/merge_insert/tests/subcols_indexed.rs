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
    let ds = nested_subset_dataset(&[0, 1, 2, 3], &[], &[]).await;
    let source = nested_subset_source();
    let when_matched = if with_condition {
        WhenMatched::update_if(&ds, "target.cells IS NULL").unwrap()
    } else {
        WhenMatched::UpdateAll
    };
    let (ds, stats) = merge_indexed(ds, source, when_matched, true).await.unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    assert_cells_filled(&ds).await;
}
/// Tests merge_insert on a table whose rows span both indexed and unindexed
/// fragments. Key 0 is in a fragment the index covers, so the plan finds it
/// through the index (the indexed take). Key 1 is in a fragment appended after
/// the index was built, so the plan finds it by scanning that fragment. The
/// target's `cells` carries the `idx`, `timestamp` and `data` children, but the
/// source's `cells` carries only `idx` and `data`. So the indexed take and the
/// unindexed scan must both emit `cells` narrowed to `idx` and `data`;
/// otherwise the plan cannot combine them.
///
/// The update must patch only the children the source supplies and keep the
/// rest: key 0 starts with null `cells`, so its `timestamp` stays null, while
/// key 1 keeps its stored `timestamp` of 42. The `update_if` condition reads
/// `target.cells`, so for key 1 it is evaluated on the row from the unindexed
/// scan.
#[tokio::test]
async fn test_indexed_partial_update_if_nested_subset_unindexed_fragment() {
    let ds = nested_subset_dataset(&[0, 2, 3], &[1, 4], &[1]).await;
    let when_matched = WhenMatched::update_if(
        &ds,
        "target.interval_idx = 0 OR array_length(target.cells) = 1",
    )
    .unwrap();
    let (ds, stats) = merge_indexed(ds, nested_subset_source(), when_matched, true)
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    assert_cells_filled(&ds).await;

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
        if key < 2 {
            let cell = cells.value(row);
            let cell = cell.as_struct();
            let idx = cell["idx"].as_primitive::<Int32Type>();
            assert_eq!(idx.value(0), key as i32 + 1, "row {row} key {key}");
            let timestamp = cell["timestamp"].as_primitive::<arrow_array::types::Int64Type>();
            if key == 0 {
                assert!(timestamp.is_null(0), "row {row} key {key}");
            } else {
                assert!(timestamp.is_valid(0), "row {row} key {key}");
                assert_eq!(timestamp.value(0), 42, "row {row} key {key}");
            }
        }
    }
}

/// Dataset keyed by `interval_idx` whose `cells` is
/// `list<struct<idx, timestamp, data>>`, with a BTree index over
/// `indexed_keys`. `unindexed_keys`, when non-empty, are appended as a
/// fragment the index does not cover. Rows keyed by `stored_cell_keys` hold
/// the single cell `{idx: -1, timestamp: 42, data: "old"}`; every other row's
/// `cells` is null.
async fn nested_subset_dataset(
    indexed_keys: &[i64],
    unindexed_keys: &[i64],
    stored_cell_keys: &[i64],
) -> Dataset {
    use arrow_array::LargeBinaryArray;
    let item_fields: arrow_schema::Fields = vec![
        Field::new("idx", DataType::Int32, true),
        Field::new("timestamp", DataType::Int64, true),
        Field::new("data", DataType::LargeBinary, true),
    ]
    .into();
    let item = Arc::new(Field::new(
        "item",
        DataType::Struct(item_fields.clone()),
        true,
    ));
    let schema = Arc::new(Schema::new(vec![
        Field::new("interval_idx", DataType::Int64, true),
        Field::new("t_start", DataType::Int64, true),
        Field::new("cells", DataType::List(item.clone()), true),
    ]));
    let target_rows = |keys: &[i64]| {
        let has_stored_cell: Vec<bool> = keys
            .iter()
            .map(|key| stored_cell_keys.contains(key))
            .collect();
        let num_stored = has_stored_cell.iter().filter(|stored| **stored).count();
        let items = StructArray::new(
            item_fields.clone(),
            vec![
                Arc::new(Int32Array::from(vec![-1; num_stored])),
                Arc::new(Int64Array::from(vec![42i64; num_stored])),
                Arc::new(LargeBinaryArray::from(vec![&b"old"[..]; num_stored])),
            ],
            None,
        );
        let cells = ListArray::new(
            item.clone(),
            OffsetBuffer::from_lengths(has_stored_cell.iter().map(|stored| *stored as usize)),
            Arc::new(items),
            Some(has_stored_cell.into()),
        );
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(keys.to_vec())),
                Arc::new(Int64Array::from(vec![0i64; keys.len()])),
                Arc::new(cells),
            ],
        )
        .unwrap()
    };
    let mut ds = Dataset::write(
        RecordBatchIterator::new([Ok(target_rows(indexed_keys))], schema.clone()),
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
    if !unindexed_keys.is_empty() {
        ds.append(
            RecordBatchIterator::new([Ok(target_rows(unindexed_keys))], schema.clone()),
            None,
        )
        .await
        .unwrap();
    }
    ds
}
/// Source for keys 0 and 1 whose `cells` carries only the `idx` and `data`
/// children: `[{idx: 1, data: "a"}]` and `[{idx: 2, data: "bb"}]`.
fn nested_subset_source() -> RecordBatch {
    use arrow_array::LargeBinaryArray;
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
    RecordBatch::try_new(
        source_schema,
        vec![Arc::new(Int64Array::from(vec![0i64, 1])), Arc::new(cells)],
    )
    .unwrap()
}
