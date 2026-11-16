// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Int32Array, NullArray, StringArray};
use arrow_schema::Field;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;

fn base(rows: usize, nullable_parent: bool) -> StructArray {
    let values: ArrayRef = Arc::new(StringArray::from_iter_values(
        (0..rows + 2).map(|i| format!("r{i}")),
    ));
    let sibling: ArrayRef = Arc::new(Int32Array::from_iter_values(
        (0..rows + 2).map(|i| i as i32),
    ));
    let mask = NullBuffer::from((0..rows + 2).map(|row| row != 1).collect::<Vec<_>>());
    let inner: ArrayRef = Arc::new(StructArray::new(
        vec![
            Arc::new(Field::new("value", DataType::Utf8, false)),
            Arc::new(Field::new("sibling", DataType::Int32, true)),
        ]
        .into(),
        vec![values, sibling.clone()],
        Some(mask.clone()),
    ));
    StructArray::new(
        vec![
            Arc::new(Field::new(
                "inner",
                inner.data_type().clone(),
                nullable_parent,
            )),
            Arc::new(Field::new("outer_sibling", DataType::Int32, true)),
        ]
        .into(),
        vec![inner, sibling],
        Some(mask),
    )
    .slice(1, rows)
}

#[rstest]
fn splice_charges_complete_spine_and_preserves_siblings(
    #[values(false, true)] nullable_parent: bool,
    #[values(10_usize, 513)] rows: usize,
) {
    let source = base(rows, nullable_parent);
    let before = source.to_data();
    let replacement: ArrayRef =
        Arc::new(StringArray::from_iter((0..rows).map(|row| {
            (![0, 1, rows - 1].contains(&row)).then_some("replacement")
        })));
    let mask_bytes = rows.div_ceil(8);
    let bytes = mask_bytes * if nullable_parent { 1 } else { 2 };
    let budget = AllocationBudget::new(bytes);
    let mut short = budget.try_reserve(bytes - 1).unwrap();
    assert!(
        try_splice_struct_in(&source, &[0, 0], replacement.clone(), &mut short)
            .unwrap()
            .is_none()
    );
    assert_eq!(short.len(), bytes - 1);
    assert_eq!(source.to_data(), before);
    drop(short);
    assert_eq!(budget.reserved(), 0);

    let mut workspace = budget.try_reserve(bytes).unwrap();
    let result = try_splice_struct_in(&source, &[0, 0], replacement.clone(), &mut workspace)
        .unwrap()
        .unwrap();
    assert!(workspace.is_empty());
    let output = result.as_struct();
    let inner = output.column(0).as_struct();
    let original_inner = source.column(0).as_struct();
    assert!(Arc::ptr_eq(inner.column(0), &replacement));
    assert!(Arc::ptr_eq(inner.column(1), original_inner.column(1)));
    assert!(Arc::ptr_eq(output.column(1), source.column(1)));
    assert_eq!(output.data_type(), source.data_type());
    for row in 0..rows {
        assert_eq!(inner.is_valid(row), ![0, 1, rows - 1].contains(&row));
        assert_eq!(
            output.is_valid(row),
            if nullable_parent {
                row != 0
            } else {
                ![0, 1, rows - 1].contains(&row)
            }
        );
    }
    assert_eq!(source.to_data(), before);
    // Thin bitmap slices keep their full charged backing, independently per level.
    let parent_alias = output.nulls().unwrap().buffer().slice_with_length(0, 1);
    let inner_alias = inner.nulls().unwrap().buffer().slice_with_length(0, 1);
    drop((result, workspace));
    assert_eq!(budget.reserved(), bytes);
    assert!(budget.try_reserve(1).is_none());
    drop(parent_alias);
    assert_eq!(budget.reserved(), mask_bytes);
    drop(inner_alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn unchanged_mask_needs_no_payload_credit(#[values(false, true)] nullable: bool) {
    let source = base(10, nullable);
    let replacement: ArrayRef = Arc::new(StringArray::from_iter(
        (0..10).map(|row| (row != 0).then_some("new")),
    ));
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let output = try_splice_struct_in(&source, &[0, 0], replacement, &mut workspace)
        .unwrap()
        .unwrap();
    let output = output.as_struct();
    assert_eq!(output.nulls(), source.nulls());
    assert_eq!(
        output.nulls().unwrap().buffer().as_ptr(),
        source.nulls().unwrap().buffer().as_ptr()
    );
    let inner = output.column(0).as_struct();
    let old_inner = source.column(0).as_struct();
    assert_eq!(inner.nulls(), old_inner.nulls());
    assert_eq!(
        inner.nulls().unwrap().buffer().as_ptr(),
        old_inner.nulls().unwrap().buffer().as_ptr()
    );
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn splice_without_source_validity(
    #[values(false, true)] is_nullable: bool,
    #[values(0_usize, 3)] rows: usize,
) {
    let source = StructArray::new(
        vec![Arc::new(Field::new("value", DataType::Utf8, is_nullable))].into(),
        vec![Arc::new(StringArray::from(vec!["old"; rows]))],
        None,
    );
    let replacement: ArrayRef = Arc::new(StringArray::from_iter(
        (0..rows).map(|row| (row != 1).then_some("new")),
    ));
    let bytes = if is_nullable { 0 } else { rows.div_ceil(8) };
    let budget = AllocationBudget::new(bytes);
    let mut workspace = budget.try_reserve(bytes).unwrap();
    let output = try_splice_struct_in(&source, &[0], replacement.clone(), &mut workspace)
        .unwrap()
        .unwrap();
    let result = output.as_struct();
    assert!(workspace.is_empty());
    assert!(source.nulls().is_none());
    assert!(Arc::ptr_eq(result.column(0), &replacement));
    assert_eq!(result.len(), rows);
    assert_eq!(result.null_count(), usize::from(!is_nullable && rows > 0));
    for row in 0..rows {
        assert_eq!(result.is_valid(row), is_nullable || row != 1);
    }
    drop((output, workspace));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::empty(&[], false, false, "nonempty child path")]
#[case::out_of_range(&[2], false, false, "child 2 exceeds 2 fields")]
#[case::not_struct(&[1, 0], false, false, "child 1 is not a Struct")]
#[case::wrong_type(&[0, 0], true, false, "replacement has type")]
#[case::wrong_length(&[0, 0], false, true, "expected Utf8 and 10 rows")]
fn invalid_splice_is_not_capacity_refusal(
    #[case] path: &[usize],
    #[case] wrong_type: bool,
    #[case] short: bool,
    #[case] message: &str,
) {
    let source = base(10, false);
    let replacement: ArrayRef = if wrong_type {
        Arc::new(Int32Array::from(vec![1; 10]))
    } else {
        Arc::new(StringArray::from(vec!["new"; if short { 9 } else { 10 }]))
    };
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let error = try_splice_struct_in(&source, path, replacement, &mut workspace).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn required_logical_nulls_do_not_allocate_outside_workspace() {
    let source = StructArray::new(
        vec![Arc::new(Field::new("null", DataType::Null, false))].into(),
        vec![Arc::new(NullArray::new(2))],
        Some(NullBuffer::new_null(2)),
    );
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let error = try_splice_struct_in(&source, &[0], Arc::new(NullArray::new(2)), &mut workspace)
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("cannot materialize logical nulls")
    );
    assert_eq!(budget.reserved(), 0);
}
