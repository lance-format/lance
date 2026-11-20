// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Int32Array, StringArray};
use arrow_schema::Field;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;

fn tree(rows: usize, is_nullable: bool) -> StructArray {
    let values: ArrayRef = Arc::new(StringArray::from(vec!["payload"; rows + 1]));
    let mask = NullBuffer::from((0..rows + 1).map(|row| row % 3 != 1).collect::<Vec<_>>());
    let child: ArrayRef = Arc::new(StructArray::new(
        vec![Arc::new(Field::new("value", DataType::Utf8, false))].into(),
        vec![values],
        Some(mask.clone()),
    ));
    StructArray::new(
        vec![
            Arc::new(Field::new("child", child.data_type().clone(), is_nullable)),
            Arc::new(Field::new("sibling", DataType::Int32, true)),
        ]
        .into(),
        vec![child, Arc::new(Int32Array::from(vec![7; rows + 1]))],
        Some(mask),
    )
    .slice(1, rows)
}

fn donor(valid: Vec<bool>) -> StructArray {
    let len = valid.len();
    StructArray::new_empty_fields(len, Some(NullBuffer::from(valid)))
}

#[rstest]
fn validity_restores_complete_tree_before_building(
    #[values(9_usize, 513)] rows: usize,
    #[values(false, true)] is_reversed: bool,
) {
    let source = tree(rows, false);
    let source_before = source.to_data();
    let validity = donor((0..rows + 2).map(|row| row % 3 != 2).collect());
    let sources = [&validity];
    // Nonzero selected offsets exercise sliced old masks and independently routed donors.
    let indices = (0..rows).map(|row| (0, row + 1)).collect::<Vec<_>>();
    let mut updates = [
        StructValiditySelection {
            path: &[],
            sources: &sources,
            indices: &indices,
        },
        StructValiditySelection {
            path: &[0],
            sources: &sources,
            indices: &indices,
        },
    ];
    if is_reversed {
        updates.reverse();
    }
    let bytes = rows.div_ceil(8);
    let budget = AllocationBudget::new(bytes * 2);
    let mut short = budget.try_reserve(bytes * 2 - 1).unwrap();
    assert!(
        try_replace_struct_validities_in(&source, &updates, &mut short)
            .unwrap()
            .is_none()
    );
    assert_eq!(short.len(), bytes * 2 - 1);
    drop(short);
    let mut workspace = budget.try_reserve(bytes * 2).unwrap();
    let output = try_replace_struct_validities_in(&source, &updates, &mut workspace)
        .unwrap()
        .unwrap();
    let outer = output.as_struct();
    let inner = outer.column(0).as_struct();
    assert!(Arc::ptr_eq(
        inner.column(0),
        source.column(0).as_struct().column(0)
    ));
    assert!(Arc::ptr_eq(outer.column(1), source.column(1)));
    for row in 0..rows {
        let expected = (row + 1) % 3 != 2;
        assert_eq!(outer.is_valid(row), expected);
        assert_eq!(inner.is_valid(row), expected);
    }
    assert!(
        source.is_null(0) && outer.is_valid(0),
        "a formerly null row must revive"
    );
    assert!(
        source.is_valid(1) && outer.is_null(1),
        "the new mask must also introduce nulls"
    );
    assert_eq!(source.to_data(), source_before);
    let outer_alias = outer.nulls().unwrap().buffer().slice_with_length(0, 1);
    let inner_alias = inner.nulls().unwrap().buffer().slice_with_length(0, 1);
    drop((output, workspace));
    assert_eq!(budget.reserved(), bytes * 2);
    drop(outer_alias);
    assert_eq!(budget.reserved(), bytes);
    assert!(budget.try_reserve(bytes * 2).is_none());
    drop(inner_alias);
    assert_eq!(budget.reserved(), 0);
    assert!(budget.try_reserve(bytes * 2).is_some());
}

#[rstest]
fn unselected_ancestor_masks_only_required_child(#[values(false, true)] is_nullable: bool) {
    let source = tree(9, is_nullable);
    let validity = donor((0..9).map(|row| row % 3 != 1).collect());
    let sources = [&validity];
    let indices = (0..9).map(|row| (0, row)).collect::<Vec<_>>();
    let updates = [StructValiditySelection {
        path: &[0],
        sources: &sources,
        indices: &indices,
    }];
    let bytes = if is_nullable { 2 } else { 4 };
    let budget = AllocationBudget::new(bytes);
    let mut workspace = budget.try_reserve(bytes).unwrap();
    let output = try_replace_struct_validities_in(&source, &updates, &mut workspace)
        .unwrap()
        .unwrap();
    for row in 0..9 {
        assert_eq!(
            output.is_valid(row),
            source.is_valid(row) && (is_nullable || row % 3 != 1)
        );
    }
    assert!(workspace.is_empty());
    drop((output, workspace));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn all_valid_and_empty_outputs_need_no_bitmap(#[values(0_usize, 9)] rows: usize) {
    let source = tree(rows, false);
    let validity = donor(vec![true; rows]);
    let sources = [&validity];
    let indices = (0..rows).map(|row| (0, row)).collect::<Vec<_>>();
    let updates = [
        StructValiditySelection {
            path: &[],
            sources: &sources,
            indices: &indices,
        },
        StructValiditySelection {
            path: &[0],
            sources: &sources,
            indices: &indices,
        },
    ];
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let output = try_replace_struct_validities_in(&source, &updates, &mut workspace)
        .unwrap()
        .unwrap();
    assert!(output.nulls().is_none());
    assert!(output.as_struct().column(0).nulls().is_none());
    assert_eq!(output.len(), rows);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn restoring_parent_without_required_child_is_an_error_before_allocation() {
    let source = tree(9, false);
    let validity = donor(vec![true; 9]);
    let sources = [&validity];
    let indices = (0..9).map(|row| (0, row)).collect::<Vec<_>>();
    let updates = [StructValiditySelection {
        path: &[],
        sources: &sources,
        indices: &indices,
    }];
    let budget = AllocationBudget::new(2);
    let mut workspace = budget.try_reserve(2).unwrap();
    let error = try_replace_struct_validities_in(&source, &updates, &mut workspace).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("exposes required child 'child' null at row 0")
    );
    assert_eq!(workspace.len(), 2);
}

#[test]
fn sibling_updates_use_independent_borrowed_routes() {
    let child = tree(3, false).column(0).clone();
    let source = StructArray::new(
        vec![
            Arc::new(Field::new("left", child.data_type().clone(), false)),
            Arc::new(Field::new("right", child.data_type().clone(), false)),
        ]
        .into(),
        vec![child.clone(), child],
        Some(NullBuffer::new_null(3)),
    );
    let first = donor(vec![true, false, true]);
    let second = donor(vec![false, true, true]);
    let sources = [&first, &second];
    let updates = [
        StructValiditySelection {
            path: &[1],
            sources: &sources,
            indices: &[(1, 1), (0, 0), (1, 0)],
        },
        StructValiditySelection {
            path: &[],
            sources: &sources,
            indices: &[(1, 2), (0, 1), (1, 0)],
        },
        StructValiditySelection {
            path: &[0],
            sources: &sources,
            indices: &[(0, 0), (0, 1), (0, 2)],
        },
    ];
    let budget = AllocationBudget::new(3);
    let mut workspace = budget.try_reserve(3).unwrap();
    let output = try_replace_struct_validities_in(&source, &updates, &mut workspace)
        .unwrap()
        .unwrap();
    let result = output.as_struct();
    assert_eq!(
        result.nulls(),
        Some(&NullBuffer::from(vec![true, false, false]))
    );
    assert_eq!(
        result.column(0).nulls(),
        Some(&NullBuffer::from(vec![true, false, true]))
    );
    assert_eq!(
        result.column(1).nulls(),
        Some(&NullBuffer::from(vec![true, true, false]))
    );
    assert!(workspace.is_empty());
    drop((output, workspace));
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn unchanged_selected_masks_keep_their_original_owners() {
    let source = tree(9, false);
    let sources = [&source];
    let indices = (0..9).map(|row| (0, row)).collect::<Vec<_>>();
    let updates = [StructValiditySelection {
        path: &[],
        sources: &sources,
        indices: &indices,
    }];
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let output = try_replace_struct_validities_in(&source, &updates, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(output.to_data(), source.to_data());
    assert_eq!(
        output.nulls().unwrap().buffer().as_ptr(),
        source.nulls().unwrap().buffer().as_ptr()
    );
    assert!(Arc::ptr_eq(output.as_struct().column(0), source.column(0)));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::source(&[], &[(1, 0)], false, "invalid source")]
#[case::row(&[], &[(0, 1)], false, "invalid source")]
#[case::length(&[], &[], false, "selections, expected")]
#[case::not_struct(&[1], &[(0, 0)], false, "cannot descend")]
#[case::missing(&[2], &[(0, 0)], false, "cannot descend")]
#[case::duplicate(&[], &[(0, 0)], true, "duplicate")]
fn invalid_routing_is_not_capacity_refusal(
    #[case] path: &[usize],
    #[case] indices: &[(usize, usize)],
    #[case] is_duplicate: bool,
    #[case] message: &str,
) {
    let source = tree(1, false);
    let validity = donor(vec![false]);
    let sources = [&validity];
    let mut updates = vec![StructValiditySelection {
        path,
        sources: &sources,
        indices,
    }];
    if is_duplicate {
        updates.push(StructValiditySelection {
            path,
            sources: &sources,
            indices,
        });
    }
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let error = try_replace_struct_validities_in(&source, &updates, &mut workspace).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(budget.reserved(), 0);
}
