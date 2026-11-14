// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, Float32Array};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use lance_core::{Error, utils::allocation::AllocationBudget};
use rstest::rstest;

use super::super::{
    CompositeRepDefUnraveler, DefinitionInterpretation as Meaning, RepDefUnraveler,
};
use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
};

#[rstest]
fn admitted_nested_validity_retains_physical_owners(#[values(0, 1, 7, 8, 65, 513)] count: usize) {
    let defs = (0..count).map(|i| (i % 3) as u16).collect::<Vec<_>>();
    let levels_bytes = count * 2;
    let bitmap_bytes = count.div_ceil(8);
    let payload_bytes = count * 4;
    let budget = AllocationBudget::new(2 * levels_bytes + 2 * bitmap_bytes + payload_bytes);
    let mut work = budget
        .try_reserve(2 * levels_bytes + 2 * bitmap_bytes + payload_bytes)
        .unwrap();
    // Real input allocation and the destination coexist during the copy.
    let mut input = work.try_allocate(levels_bytes).unwrap().unwrap();
    input.as_mut().copy_from_slice(bytemuck::cast_slice(&defs));
    let source = RepDefUnraveler::try_new_in(
        None,
        Some(bytemuck::cast_slice(input.as_ref())),
        Arc::from([
            Meaning::NullableItem,
            Meaning::NullableItem,
            Meaning::AllValidItem,
        ]),
        count as u64,
        &mut work,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        budget.reserved(),
        2 * levels_bytes + 2 * bitmap_bytes + payload_bytes
    );
    drop(input);
    let mut composite = CompositeRepDefUnraveler::new(vec![source]);
    if bitmap_bytes > 0 {
        let mut short = work.try_split(bitmap_bytes - 1).unwrap();
        assert!(
            composite
                .try_unravel_validity_in(count, &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(composite.unravelers[0].current_layer, 0);
        assert_eq!(short.len(), bitmap_bytes - 1);
        drop(short);
        drop(work);
        work = budget
            .try_reserve(2 * bitmap_bytes + payload_bytes)
            .unwrap();
    }
    let inner = composite
        .try_unravel_validity_in(count, &mut work)
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        inner.iter().collect::<Vec<_>>(),
        (0..count).map(|i| i % 3 == 0).collect::<Vec<_>>()
    );
    let outer = composite
        .try_unravel_validity_in(count, &mut work)
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        outer.iter().collect::<Vec<_>>(),
        (0..count).map(|i| i % 3 != 2).collect::<Vec<_>>()
    );
    assert!(
        composite
            .try_unravel_validity_in(count, &mut work)
            .unwrap()
            .unwrap()
            .is_none()
    );
    let mut values = work.try_allocate(payload_bytes).unwrap().unwrap();
    for (i, slot) in values
        .as_mut()
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .enumerate()
    {
        slot.copy_from_slice(&(i as f32).to_le_bytes());
    }
    let data = DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::from(Buffer::from(values.freeze())),
        num_values: count as u64,
        bits_per_value: 32,
        block_info: BlockInfo::new(),
    });
    let actual = data
        .try_to_arrow_in(&DataType::Float32, &mut work)
        .unwrap()
        .unwrap()
        .into_builder()
        .nulls(Some(inner.clone()))
        .build()
        .unwrap();
    let expected = Float32Array::from(
        (0..count)
            .map(|i| (i % 3 == 0).then_some(i as f32))
            .collect::<Vec<_>>(),
    );
    assert_eq!(actual, expected.to_data());
    let last_bitmap = inner.slice(0, count.min(1));
    drop(inner);
    drop(outer);
    drop(data);
    drop(composite);
    drop(work);
    assert_eq!(budget.reserved(), payload_bytes + bitmap_bytes);
    drop(actual);
    assert_eq!(budget.reserved(), bitmap_bytes);
    drop(last_bitmap);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn logical_compaction_keeps_both_level_allocations_charged() {
    let budget = AllocationBudget::new(25);
    let mut work = budget.try_reserve(25).unwrap();
    let mut source = RepDefUnraveler::try_new_in(
        Some(&[1, 1, 0, 0, 1, 1]),
        Some(&[0, 0, 0, 0, 1, 1]),
        Arc::from([Meaning::NullableItem, Meaning::NullableList]),
        6,
        &mut work,
    )
    .unwrap()
    .unwrap();
    source.decimate(2).unwrap();
    assert_eq!(source.rep_levels.as_deref().unwrap(), &[1, 0, 1]);
    assert_eq!(source.def_levels.as_deref().unwrap(), &[0, 0, 1]);
    assert_eq!(work.len(), 1);
    assert!(budget.try_reserve(1).is_none());
    let mut composite = CompositeRepDefUnraveler::new(vec![source]);
    let validity = composite
        .try_unravel_validity_in(3, &mut work)
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(validity.iter().collect::<Vec<_>>(), [true, true, false]);
    // Compatibility only: the ordinary offset builder remains outside admission.
    let (offsets, list_validity) = composite.unravel_offsets::<i32>().unwrap();
    assert_eq!(offsets.as_ref(), &[0, 2, 3]);
    assert_eq!(
        list_validity.unwrap().iter().collect::<Vec<_>>(),
        [true, true]
    );
    assert_eq!(budget.reserved(), 25);
    drop(composite);
    assert_eq!(budget.reserved(), 1);
    drop(validity);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::matching(false)]
#[case::mismatched(true)]
fn sibling_checks_are_atomic_without_temporary_bitmaps(#[case] mismatch: bool) {
    let budget = AllocationBudget::new(25);
    let mut work = budget.try_reserve(25).unwrap();
    let nullable = Arc::from([Meaning::NullableItem]);
    let left = RepDefUnraveler::try_new_in(None, Some(&[0, 1, 0, 1]), nullable, 4, &mut work)
        .unwrap()
        .unwrap();
    let right =
        RepDefUnraveler::try_new_in(None, None, Arc::from([Meaning::NullableItem]), 4, &mut work)
            .unwrap()
            .unwrap();
    let mut composite = CompositeRepDefUnraveler::new(vec![left, right]);
    let other_defs = if mismatch {
        [0, 1, 0, 1, 0, 0, 0, 1]
    } else {
        [0, 1, 0, 1, 0, 0, 0, 0]
    };
    let other = RepDefUnraveler::try_new_in(
        None,
        Some(&other_defs),
        Arc::from([Meaning::NullableItem]),
        8,
        &mut work,
    )
    .unwrap()
    .unwrap();
    composite.add_compatibility_check(CompositeRepDefUnraveler::new(vec![other]));
    if mismatch {
        let error = composite.try_unravel_validity_in(8, &mut work).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("incompatible validity"));
        assert_eq!(work.len(), 1);
        assert_eq!(composite.unravelers[0].current_layer, 0);
        assert_eq!(composite.comparisons[0].unravelers[0].current_layer, 0);
        // A corrected producer may retry; rejection did not consume the base layer.
        composite.comparisons.clear();
    }
    let actual = composite
        .try_unravel_validity_in(8, &mut work)
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        actual.iter().collect::<Vec<_>>(),
        [true, false, true, false, true, true, true, true]
    );
    assert!(work.is_empty());
    drop(composite);
    drop(work);
    assert_eq!(budget.reserved(), 1);
    drop(actual);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn list_specials_are_not_leaf_values() {
    let budget = AllocationBudget::new(25);
    let mut work = budget.try_reserve(25).unwrap();
    let source = RepDefUnraveler::try_new_in(
        Some(&[1, 0, 1, 1, 1, 1]),
        Some(&[0, 1, 2, 3, 4, 0]),
        Arc::from([
            Meaning::NullableItem,
            Meaning::NullableAndEmptyableList,
            Meaning::NullableItem,
        ]),
        3,
        &mut work,
    )
    .unwrap()
    .unwrap();
    let mut composite = CompositeRepDefUnraveler::new(vec![source]);
    let error = composite.try_unravel_validity_in(6, &mut work).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("produced 3 values, expected 6"));
    assert_eq!(work.len(), 1);
    let values = composite
        .try_unravel_validity_in(3, &mut work)
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(values.iter().collect::<Vec<_>>(), [true, false, true]);
    let error = composite.try_unravel_validity_in(5, &mut work).unwrap_err();
    assert!(error.to_string().contains("unravel list offsets"));
    assert_eq!(composite.unravelers[0].current_layer, 1);
}

#[rstest]
#[case::different_lengths(Some(vec![1,0]), Some(vec![0]), "lengths differ")]
#[case::bad_rep(Some(vec![2]), Some(vec![0]), "level exceeds")]
#[case::bad_def(Some(vec![1]), Some(vec![3]), "level exceeds")]
fn malformed_levels_fail_before_allocation(
    #[case] rep: Option<Vec<u16>>,
    #[case] def: Option<Vec<u16>>,
    #[case] message: &str,
) {
    let budget = AllocationBudget::new(32);
    let mut work = budget.try_reserve(32).unwrap();
    let error = RepDefUnraveler::try_new_in(
        rep.as_deref(),
        def.as_deref(),
        Arc::from([Meaning::NullableItem, Meaning::NullableList]),
        1,
        &mut work,
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message));
    assert_eq!(work.len(), 32);
}

#[test]
fn whole_level_pair_refuses_without_consuming_credit() {
    let budget = AllocationBudget::new(11);
    let mut work = budget.try_reserve(11).unwrap();
    assert!(
        RepDefUnraveler::try_new_in(
            Some(&[1, 0, 1]),
            Some(&[0, 1, 0]),
            Arc::from([Meaning::NullableItem, Meaning::NullableList]),
            3,
            &mut work
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(work.len(), 11);
    assert_eq!(budget.reserved(), 11);
}

#[test]
fn all_valid_comparison_does_not_need_bitmap_storage() {
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    let mut composite = CompositeRepDefUnraveler::new(vec![
        RepDefUnraveler::try_new_in(
            None,
            None,
            Arc::from([Meaning::AllValidItem]),
            513,
            &mut work,
        )
        .unwrap()
        .unwrap(),
    ]);
    assert!(
        composite
            .try_unravel_validity_in(513, &mut work)
            .unwrap()
            .unwrap()
            .is_none()
    );
    assert_eq!(budget.reserved(), 0);
}
