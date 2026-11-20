// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::OffsetSizeTrait;
use arrow_buffer::{NullBuffer, OffsetBuffer};
use lance_core::{
    Error,
    utils::allocation::{AllocationBudget, Reservation},
};
use rstest::rstest;

use super::{CompositeRepDefUnraveler, Meaning, RepDefUnraveler};
use crate::repdef::RepDefBuilder;

fn pages(
    rep: &[u16],
    def: Option<&[u16]>,
    meanings: &[Meaning],
    cuts: &[usize],
    work: &mut Reservation,
) -> CompositeRepDefUnraveler {
    let mut start = 0;
    let sources = cuts
        .iter()
        .map(|&end| {
            let source = RepDefUnraveler::try_new_in(
                Some(&rep[start..end]),
                def.map(|d| &d[start..end]),
                Arc::from(meanings),
                (end - start) as u64,
                work,
            )
            .unwrap()
            .unwrap();
            start = end;
            source
        })
        .collect();
    assert_eq!(start, rep.len());
    CompositeRepDefUnraveler::new(sources)
}

fn check_lists<T: OffsetSizeTrait + bytemuck::Pod>(kind: usize) {
    // Independent logical outputs, including a selection starting inside a list.
    let (rep, def, meaning, expected, valid): (_, _, _, _, Option<Vec<bool>>) = match kind {
        0 => (
            vec![1, 0, 1],
            None,
            Meaning::AllValidList,
            vec![0, 2, 3],
            None,
        ),
        1 => (
            vec![1, 1, 1],
            Some(vec![1, 1, 1]),
            Meaning::NullableList,
            vec![0, 0, 0, 0],
            Some(vec![false; 3]),
        ),
        2 => (
            vec![1, 1, 1],
            Some(vec![1, 1, 1]),
            Meaning::EmptyableList,
            vec![0, 0, 0, 0],
            None,
        ),
        3 => (
            vec![1, 0, 1, 1, 1, 0],
            Some(vec![0, 0, 1, 2, 0, 0]),
            Meaning::NullableAndEmptyableList,
            vec![0, 2, 2, 2, 4],
            Some(vec![true, false, true, true]),
        ),
        4 => (
            vec![],
            Some(vec![]),
            Meaning::NullableAndEmptyableList,
            vec![0],
            Some(vec![]),
        ),
        5 => (
            vec![0, 0, 1, 0],
            None,
            Meaning::AllValidList,
            vec![2, 4],
            None,
        ),
        _ => unreachable!(),
    };
    let levels = (rep.len() + def.as_ref().map_or(0, Vec::len)) * 2;
    let offsets_bytes = expected.len() * std::mem::size_of::<T>();
    let bitmap_bytes = valid.as_ref().map_or(0, |v| v.len().div_ceil(8));
    let output = offsets_bytes + bitmap_bytes;
    // Exercise every split, including empty pages and continuation-only pages.
    for cut in 0..=rep.len() {
        let budget = AllocationBudget::new(levels * 2 + output);
        let mut work = budget.try_reserve(levels * 2 + output).unwrap();
        let mut source = pages(
            &rep,
            def.as_deref(),
            &[meaning],
            &[cut, rep.len()],
            &mut work,
        );
        let comparison = pages(&rep, def.as_deref(), &[meaning], &[rep.len()], &mut work);
        source.add_compatibility_check(comparison);
        let before = format!("{source:?}");
        let mut short = work.try_split(output - 1).unwrap();
        assert!(
            source
                .try_unravel_offsets_in::<T>(&mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), output - 1);
        assert_eq!(format!("{source:?}"), before);
        drop(short);
        drop(work);
        let mut work = budget.try_reserve(output).unwrap();
        let (offsets, nulls) = source
            .try_unravel_offsets_in::<T>(&mut work)
            .unwrap()
            .unwrap();
        assert!(work.is_empty());
        assert_eq!(
            offsets.iter().map(|v| v.as_usize()).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(nulls.as_ref().map(|v| v.iter().collect::<Vec<_>>()), valid);
        assert_eq!(
            source.unravelers.iter().map(|s| s.num_items).sum::<u64>(),
            (expected.len() - 1) as u64
        );
        // Compaction never refunds a surviving physical input allocation.
        drop(work);
        assert_eq!(budget.reserved(), levels * 2 + output);
        let alias = offsets.slice(0, 0);
        drop(offsets);
        drop(source);
        assert_eq!(budget.reserved(), output);
        drop(nulls);
        assert_eq!(budget.reserved(), offsets_bytes);
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
fn list_outputs_and_atomic_shortage(
    #[values(0, 1, 2, 3, 4, 5)] kind: usize,
    #[values(false, true)] large: bool,
) {
    if large {
        check_lists::<i64>(kind);
    } else {
        check_lists::<i32>(kind);
    }
}

#[rstest]
fn nested_lists_across_every_page_boundary(#[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)] cut: usize) {
    // Three rows of List<List<i32>>, inner lengths [1,2,2,2,2].
    let rep = [2, 1, 0, 2, 0, 2, 0, 1, 0];
    let budget = AllocationBudget::new(18 + 6 * 4 + 4 * 8);
    let mut work = budget.try_reserve(18 + 6 * 4 + 4 * 8).unwrap();
    let mut source = pages(
        &rep,
        None,
        &[Meaning::AllValidList, Meaning::AllValidList],
        &[cut, 9],
        &mut work,
    );
    let (inner, validity) = source
        .try_unravel_offsets_in::<i32>(&mut work)
        .unwrap()
        .unwrap();
    assert_eq!(inner.as_ref(), &[0, 1, 3, 5, 7, 9]);
    assert!(validity.is_none());
    let (outer, validity) = source
        .try_unravel_offsets_in::<i64>(&mut work)
        .unwrap()
        .unwrap();
    assert_eq!(outer.as_ref(), &[0, 2, 3, 5]);
    assert!(validity.is_none());
    assert_eq!(budget.reserved(), 18 + 6 * 4 + 4 * 8);
    assert!(work.is_empty());
    drop(source);
    drop(inner);
    drop(outer);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn nullable_structs_inside_null_and_empty_lists() {
    // [null, [null, {child:null}, {child:valid}], []]
    let rep = [1, 1, 0, 0, 1];
    let def = [3, 2, 1, 0, 4];
    let budget = AllocationBudget::new(20 + 2 + 17);
    let mut work = budget.try_reserve(39).unwrap();
    let mut source = pages(
        &rep,
        Some(&def),
        &[
            Meaning::NullableItem,
            Meaning::NullableItem,
            Meaning::NullableAndEmptyableList,
        ],
        &[2, 5],
        &mut work,
    );
    let leaf = source
        .try_unravel_validity_in(3, &mut work)
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(leaf.iter().collect::<Vec<_>>(), [false, false, true]);
    let parent = source
        .try_unravel_validity_in(3, &mut work)
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(parent.iter().collect::<Vec<_>>(), [false, true, true]);
    let (offsets, validity) = source
        .try_unravel_offsets_in::<i32>(&mut work)
        .unwrap()
        .unwrap();
    assert_eq!(offsets.as_ref(), &[0, 0, 3, 3]);
    assert_eq!(
        validity.unwrap().iter().collect::<Vec<_>>(),
        [false, true, true]
    );
    assert!(work.is_empty());
}

#[test]
fn nested_null_and_empty_lists_from_encoded_structure() {
    // [null, [], [null, [], [1, null, 2]], [[]]]
    let mut builder = RepDefBuilder::default();
    builder.add_offsets(
        OffsetBuffer::new(vec![0_i32, 0, 0, 3, 4].into()),
        Some(NullBuffer::from(vec![false, true, true, true])),
    );
    builder.add_offsets(
        OffsetBuffer::new(vec![0_i32, 0, 0, 3, 3].into()),
        Some(NullBuffer::from(vec![false, true, true, true])),
    );
    builder.add_validity_bitmap(NullBuffer::from(vec![true, false, true]));
    let encoded = RepDefBuilder::serialize(vec![builder]);
    let rep = encoded.repetition_levels.unwrap();
    let def = encoded.definition_levels.unwrap();
    for cut in 0..=rep.len() {
        let budget = AllocationBudget::new(512);
        let mut work = budget.try_reserve(512).unwrap();
        let mut source = pages(
            &rep,
            Some(&def),
            &encoded.def_meaning,
            &[cut, rep.len()],
            &mut work,
        );
        let leaf = source
            .try_unravel_validity_in(3, &mut work)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(leaf.iter().collect::<Vec<_>>(), [true, false, true]);
        let (inner, inner_valid) = source
            .try_unravel_offsets_in::<i32>(&mut work)
            .unwrap()
            .unwrap();
        assert_eq!(inner.as_ref(), &[0, 0, 0, 3, 3]);
        assert_eq!(
            inner_valid.unwrap().iter().collect::<Vec<_>>(),
            [false, true, true, true]
        );
        let (outer, outer_valid) = source
            .try_unravel_offsets_in::<i64>(&mut work)
            .unwrap()
            .unwrap();
        assert_eq!(outer.as_ref(), &[0, 0, 0, 3, 4]);
        assert_eq!(
            outer_valid.unwrap().iter().collect::<Vec<_>>(),
            [false, true, true, true]
        );
        drop(source);
        drop(work);
        assert_eq!(budget.reserved(), 1 + 5 * 4 + 5 * 8);
        drop(leaf);
        drop(inner);
        drop(outer);
        assert_eq!(budget.reserved(), 0);
    }
}

#[test]
fn equal_offsets_do_not_hide_sibling_null_disagreement() {
    let budget = AllocationBudget::new(64);
    let mut work = budget.try_reserve(64).unwrap();
    let meanings = [Meaning::NullableAndEmptyableList];
    let mut source = pages(&[1, 1], Some(&[1, 2]), &meanings, &[2], &mut work);
    let sibling = pages(&[1, 1], Some(&[2, 1]), &meanings, &[1, 2], &mut work);
    source.add_compatibility_check(sibling);
    let before = format!("{source:?}");
    let credit = work.len();
    let error = source.try_unravel_offsets_in::<i64>(&mut work).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("incompatible list metadata"));
    assert_eq!(format!("{source:?}"), before);
    assert_eq!(work.len(), credit);
}

#[rstest]
#[case::offsets(0)]
#[case::validity(1)]
#[case::trailing_children(2)]
#[case::recursive_sibling(3)]
fn sibling_disagreement_preserves_all_inputs(#[case] kind: usize) {
    let budget = AllocationBudget::new(256);
    let mut work = budget.try_reserve(256).unwrap();
    let mut source = pages(
        &[1, 0, 1],
        Some(&[0, 0, 0]),
        &[Meaning::NullableList],
        &[3],
        &mut work,
    );
    let (rep, def) = match kind {
        0 | 3 => (vec![1, 1, 0], vec![0, 0, 0]),
        1 => (vec![1, 0, 1], vec![0, 0, 1]),
        2 => (vec![1, 0, 1, 0], vec![0, 0, 0, 0]),
        _ => unreachable!(),
    };
    let mismatch = pages(
        &rep,
        Some(&def),
        &[Meaning::NullableList],
        &[rep.len()],
        &mut work,
    );
    if kind == 3 {
        let mut sibling = pages(
            &[1, 0, 1],
            Some(&[0, 0, 0]),
            &[Meaning::NullableList],
            &[3],
            &mut work,
        );
        sibling.add_compatibility_check(mismatch);
        source.add_compatibility_check(sibling);
    } else {
        source.add_compatibility_check(mismatch);
    }
    let before = format!("{source:?}");
    let credit = work.len();
    let error = source.try_unravel_offsets_in::<i32>(&mut work).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("incompatible list metadata"));
    assert_eq!(format!("{source:?}"), before);
    assert_eq!(work.len(), credit);
    drop(source);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn absent_bitmap_compares_as_all_valid_under_null_parent() {
    let budget = AllocationBudget::new(64);
    let mut work = budget.try_reserve(64).unwrap();
    // A parent null masks this nonnullable child list. Its own bitmap is absent.
    let mut source = pages(
        &[1],
        Some(&[1]),
        &[Meaning::AllValidList, Meaning::NullableItem],
        &[1],
        &mut work,
    );
    let sibling = pages(
        &[1],
        Some(&[1]),
        &[Meaning::EmptyableList, Meaning::AllValidItem],
        &[1],
        &mut work,
    );
    source.add_compatibility_check(sibling);
    let (offsets, validity) = source
        .try_unravel_offsets_in::<i32>(&mut work)
        .unwrap()
        .unwrap();
    assert_eq!(offsets.as_ref(), &[0, 0]);
    assert!(validity.is_none());
    let parent = source.try_unravel_validity_in(1, &mut work).unwrap_err();
    assert!(matches!(parent, Error::InvalidInput { .. }));
    assert!(parent.to_string().contains("incompatible validity"));
}

#[rstest]
fn malformed_list_state_is_atomic(#[values(0, 1, 2, 3)] kind: usize) {
    let budget = AllocationBudget::new(64);
    let mut work = budget.try_reserve(64).unwrap();
    let mut source = pages(
        &[1, 0],
        Some(&[0, 0]),
        &[Meaning::NullableList],
        &[2],
        &mut work,
    );
    let page = &mut source.unravelers[0];
    match kind {
        0 => page.rep_levels.as_mut().unwrap()[0] = 2,
        1 => page.def_levels.as_mut().unwrap()[0] = 2,
        2 => page.current_layer = 1,
        3 => page.rep_levels = None,
        _ => unreachable!(),
    }
    let before = format!("{source:?}");
    let credit = work.len();
    let error = source.try_unravel_offsets_in::<i32>(&mut work).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("list") || error.to_string().contains("offsets"));
    assert_eq!(format!("{source:?}"), before);
    assert_eq!(work.len(), credit);
}
