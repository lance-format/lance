// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use arrow_array::{BinaryArray, Int32Array, LargeBinaryArray, ListArray, StringArray};
use arrow_schema::Field;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;

fn source() -> StructArray {
    let required: ArrayRef = Arc::new(StringArray::from(vec![
        Some("discard"),
        None,
        Some("b"),
        Some("hidden"),
        Some(""),
    ]));
    let leaf: ArrayRef = Arc::new(BinaryArray::from(vec![
        Some(b"a".as_slice()),
        None,
        Some(b"bc".as_slice()),
        Some(b"".as_slice()),
        Some([255_u8, 0].as_slice()),
    ]));
    let lists: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("frame", DataType::Binary, true)),
        OffsetBuffer::new(vec![0, 1, 1, 3, 4, 5].into()),
        leaf.clone(),
        Some(NullBuffer::from(vec![true, false, true, true, true])),
    ));
    let inner: ArrayRef = Arc::new(StructArray::new(
        vec![Arc::new(Field::new("bytes", DataType::Binary, true))].into(),
        vec![leaf],
        Some(NullBuffer::from(vec![true, false, true, true, true])),
    ));
    let arrays = vec![required, lists, inner];
    let fields = arrays
        .iter()
        .enumerate()
        .map(|(index, array)| {
            Arc::new(
                Field::new(
                    format!("child{index}"),
                    array.data_type().clone(),
                    index != 0,
                )
                .with_metadata(HashMap::from([("unit".into(), "frames".into())])),
            )
        })
        .collect::<Vec<_>>();
    StructArray::new(
        fields.into(),
        arrays,
        Some(NullBuffer::from(vec![true, false, true, false, true])),
    )
    .slice(1, 4)
}

#[rstest]
#[case::struct_only(None)]
#[case::list(Some(false))]
#[case::large_list(Some(true))]
fn struct_list_composition_matches_arrow(
    #[case] outer: Option<bool>,
    #[values(false, true)] empty: bool,
) {
    let mut source: ArrayRef = Arc::new(source());
    if let Some(large) = outer {
        let field = Arc::new(Field::new("item", source.data_type().clone(), true));
        let nulls = Some(NullBuffer::from(vec![true, false, true, true]));
        source = if large {
            Arc::new(GenericListArray::<i64>::new(
                field,
                OffsetBuffer::new(vec![0_i64, 1, 1, 3, 4].into()),
                source,
                nulls,
            ))
        } else {
            Arc::new(ListArray::new(
                field,
                OffsetBuffer::new(vec![0_i32, 1, 1, 3, 4].into()),
                source,
                nulls,
            ))
        };
    }
    let second = source.slice(1, 3);
    let indices = if empty {
        &[][..]
    } else {
        &[(0, 3), (1, 0), (0, 0), (0, 2), (1, 1), (0, 3)][..]
    };
    let arrays = [source.as_ref(), second.as_ref()];
    let expected = arrow_select::interleave::interleave(&arrays, indices).unwrap();
    let budget = AllocationBudget::new(4096);
    let mut workspace = budget.try_reserve(4096).unwrap();
    let actual = try_interleave_nested_binary_in(&arrays, indices, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(actual.to_data(), expected.to_data());
    assert_eq!(actual.data_type(), source.data_type());
    drop((actual, workspace));
    assert_eq!(budget.reserved(), 0);
}

fn exact_source() -> StructArray {
    let key: ArrayRef = Arc::new(StringArray::from(vec![None, Some("ab"), Some("c")]));
    let value: ArrayRef = Arc::new(LargeBinaryArray::from(vec![
        Some(b"x".as_slice()),
        None,
        Some(b"".as_slice()),
    ]));
    StructArray::new(
        vec![
            Arc::new(Field::new("key", DataType::Utf8, false)),
            Arc::new(Field::new("value", DataType::LargeBinary, true)),
        ]
        .into(),
        vec![key, value],
        Some(NullBuffer::from(vec![false, true, true])),
    )
}

#[rstest]
#[case::none(0)]
#[case::parent_only(1)]
#[case::one_short(54)]
fn struct_refuses_entire_tree_before_allocation(#[case] bytes: usize) {
    let source = exact_source();
    // Parent bitmap:1. Key:16 offsets +3 values +1 validity. Value:32+1+1.
    let budget = AllocationBudget::new(55);
    let mut workspace = budget.try_reserve(bytes).unwrap();
    assert!(
        try_interleave_nested_binary_in(&[&source], &[(0, 2), (0, 0), (0, 1)], &mut workspace)
            .unwrap()
            .is_none()
    );
    assert_eq!(workspace.len(), bytes);
    assert_eq!(budget.reserved(), bytes);
}

#[test]
fn struct_aliases_retain_each_complete_backing() {
    let source = exact_source();
    let indices = [(0, 2), (0, 0), (0, 1)];
    let budget = AllocationBudget::new(55);
    let mut workspace = budget.try_reserve(55).unwrap();
    let output = try_interleave_nested_binary_in(&[&source], &indices, &mut workspace)
        .unwrap()
        .unwrap();
    assert!(workspace.is_empty());
    assert_eq!(
        output.to_data(),
        arrow_select::interleave::interleave(&[&source], &indices)
            .unwrap()
            .to_data()
    );
    let structs = output.as_struct();
    let parent = structs.nulls().unwrap().buffer().clone();
    let key = structs
        .column(0)
        .as_string::<i32>()
        .values()
        .slice_with_length(0, 1);
    let value = structs
        .column(1)
        .as_binary::<i64>()
        .values()
        .slice_with_length(0, 1);
    drop((output, workspace));
    assert_eq!(budget.reserved(), 55);
    assert!(budget.try_reserve(1).is_none());
    drop(parent);
    assert_eq!(budget.reserved(), 54);
    drop(key);
    assert_eq!(budget.reserved(), 34);
    drop(value);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn empty_struct_keeps_length_and_nulls(
    #[values(false, true)] nullable: bool,
    #[values(4_usize, 513)] rows: usize,
) {
    let nulls = nullable.then(|| NullBuffer::from(vec![true, false, true]));
    let source = StructArray::new_empty_fields(3, nulls);
    let indices = (0..rows).map(|row| (0, row % 3)).collect::<Vec<_>>();
    // Cross Arrow's 64-byte growth boundary without rounding the charged bitmap.
    let bytes = if nullable { rows.div_ceil(8) } else { 0 };
    let budget = AllocationBudget::new(bytes);
    let mut workspace = budget.try_reserve(bytes).unwrap();
    let output = try_interleave_nested_binary_in(&[&source], &indices, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(output.len(), rows);
    assert!(workspace.is_empty());
    assert_eq!(budget.reserved(), bytes);
    assert_eq!(
        output.to_data(),
        arrow_select::interleave::interleave(&[&source], &indices)
            .unwrap()
            .to_data()
    );
    drop((output, workspace));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::source(1, 0, "source 1 exceeds 1 sources")]
#[case::row(0, 3, "row 3 exceeds source 0 length 3")]
fn empty_struct_validates_indices(
    #[case] source_index: usize,
    #[case] row: usize,
    #[case] message: &str,
) {
    let source = StructArray::new_empty_fields(3, None);
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let error = try_interleave_nested_binary_in(&[&source], &[(source_index, row)], &mut workspace)
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message));
}

#[test]
fn unsupported_struct_child_is_an_error_for_empty_selection() {
    let source = StructArray::new(
        vec![Arc::new(Field::new("child", DataType::Int32, false))].into(),
        vec![Arc::new(Int32Array::from(vec![1]))],
        None,
    );
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let error = try_interleave_nested_binary_in(&[&source], &[], &mut workspace).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("does not support Int32"));
    assert_eq!(budget.reserved(), 0);
}
