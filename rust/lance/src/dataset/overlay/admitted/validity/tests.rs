// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{StringArray, UInt32Array, cast::AsArray};
use arrow_schema::{DataType, Field as ArrowField};
use lance_core::utils::allocation::AllocationBudget;
use lance_encoding::data::try_replace_struct_validities_in;
use rstest::rstest;

use super::*;

mod file;

fn validity(values: Vec<bool>) -> StructArray {
    let len = values.len();
    StructArray::try_new_with_length(
        Default::default(),
        vec![],
        Some(NullBuffer::from(values)),
        len,
    )
    .unwrap()
}

#[rstest]
#[case::empty(vec![])]
#[case::range((0..9).collect())]
#[case::duplicates(vec![6, 0, 2, 6, 1, 4, 3, 5, 8, 7])]
fn global_precedence_is_independent_of_field_order(
    #[case] rows: Vec<u32>,
    #[values(false, true)] reverse: bool,
) {
    let base = validity(rows.iter().map(|row| row.is_multiple_of(2)).collect());
    let a: [RoaringBitmap; 2] = [
        [0, 2, 4, 6].into_iter().collect(),
        [1, 2, 3, 5].into_iter().collect(),
    ];
    let b: [RoaringBitmap; 2] = [
        [2, 3, 6].into_iter().collect(),
        [0, 5, 7].into_iter().collect(),
    ];
    let ranks = [[4_usize, 8], [1, 7]];
    let coverages = [&a, &b];
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let selection = OverlayBatchSelection::Indices(UInt32Array::from(rows.clone()));
    let routings = coverages.map(|coverage| {
        AdmittedOverlayRouting::try_new(&selection, &[&coverage[0], &coverage[1]], &mut work)
            .unwrap()
            .unwrap()
    });
    let fetched = (0..2)
        .map(|field| {
            (0..2)
                .map(|source| {
                    let offsets = routings[field].fetch_offsets(source).unwrap();
                    validity(
                        offsets
                            .iter()
                            .map(|offset| {
                                let row = coverages[field][source].select(*offset).unwrap();
                                !(row as usize + ranks[field][source]).is_multiple_of(3)
                            })
                            .collect(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let sources = fetched
        .iter()
        .map(|arrays| arrays.iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut inputs = (0..2)
        .map(|field| OverlayStructValidityInput {
            routing: &routings[field],
            file_order: &ranks[field],
            sources: &sources[field],
        })
        .collect::<Vec<_>>();
    if reverse {
        inputs.reverse();
    }
    let before = work.len();
    let resolved = AdmittedStructValidityRouting::try_new(&base, &inputs, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(
        before - work.len(),
        rows.len() * size_of::<(usize, usize)>()
    );
    let resolved = resolved.selection(&[]);
    for (position, row) in rows.iter().enumerate() {
        // Reference the physical coverage and global file ranks directly, without
        // consulting either admitted routing or the proposed winner table.
        let newest = (0..2)
            .flat_map(|field| (0..2).map(move |source| (field, source)))
            .filter(|&(field, source)| coverages[field][source].contains(*row))
            .min_by_key(|&(field, source)| ranks[field][source]);
        let expected = newest.map_or(row.is_multiple_of(2), |(field, source)| {
            !(*row as usize + ranks[field][source]).is_multiple_of(3)
        });
        let (source, offset) = resolved.indices[position];
        assert_eq!(
            resolved.sources[source].is_valid(offset),
            expected,
            "physical row {row}"
        );
    }
}

#[test]
fn validity_table_refuses_before_allocation_and_retains_full_charge() {
    let base = validity(vec![false; 9]);
    let budget = AllocationBudget::new(9 * size_of::<(usize, usize)>());
    let mut short = budget
        .try_reserve(9 * size_of::<(usize, usize)>() - 1)
        .unwrap();
    let before = short.len();
    assert!(
        AdmittedStructValidityRouting::try_new(&base, &[], &mut short)
            .unwrap()
            .is_none()
    );
    assert_eq!(short.len(), before);
    drop(short);
    let mut exact = budget.try_reserve(before + 1).unwrap();
    let resolved = AdmittedStructValidityRouting::try_new(&base, &[], &mut exact)
        .unwrap()
        .unwrap();
    assert!(exact.is_empty());
    let alias = resolved.storage.slice_with_length(0, 1);
    drop((resolved, exact));
    assert_eq!(budget.reserved(), before + 1);
    assert!(budget.try_reserve(1).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    assert!(budget.try_reserve(before + 1).is_some());
}

#[rstest]
fn equal_file_rank_keeps_first_even_at_max(#[values(false, true)] first_valid: bool) {
    let base = validity(vec![false]);
    let first = validity(vec![first_valid]);
    let second = validity(vec![!first_valid]);
    let coverage: RoaringBitmap = [0].into_iter().collect();
    let budget = AllocationBudget::new(4096);
    let mut work = budget.try_reserve(4096).unwrap();
    let routing = AdmittedOverlayRouting::try_new(
        &OverlayBatchSelection::Range(0..1),
        &[&coverage],
        &mut work,
    )
    .unwrap()
    .unwrap();
    let inputs = [
        OverlayStructValidityInput {
            routing: &routing,
            file_order: &[usize::MAX],
            sources: &[&first],
        },
        OverlayStructValidityInput {
            routing: &routing,
            file_order: &[usize::MAX],
            sources: &[&second],
        },
    ];
    let resolved = AdmittedStructValidityRouting::try_new(&base, &inputs, &mut work)
        .unwrap()
        .unwrap();
    let selection = resolved.selection(&[]);
    let (source, row) = selection.indices[0];
    assert_eq!(selection.sources[source].is_valid(row), first_valid);
}

#[test]
fn invalid_validity_inputs_precede_capacity_refusal() {
    let base = validity(vec![true, true]);
    let donor = validity(vec![false]);
    let short = validity(vec![]);
    let coverage: RoaringBitmap = [0].into_iter().collect();
    let other: RoaringBitmap = [1].into_iter().collect();
    let budget = AllocationBudget::new(4096);
    let mut work = budget.try_reserve(4096).unwrap();
    let route = AdmittedOverlayRouting::try_new(
        &OverlayBatchSelection::Range(0..2),
        &[&coverage, &other],
        &mut work,
    )
    .unwrap()
    .unwrap();
    drop(work);
    let mut zero = budget.try_reserve(0).unwrap();
    for input in [
        OverlayStructValidityInput {
            routing: &route,
            file_order: &[0],
            sources: &[&donor],
        },
        OverlayStructValidityInput {
            routing: &route,
            file_order: &[0],
            sources: &[&donor, &donor],
        },
        OverlayStructValidityInput {
            routing: &route,
            file_order: &[1, 0],
            sources: &[&donor, &donor],
        },
        OverlayStructValidityInput {
            routing: &route,
            file_order: &[0, 0],
            sources: &[&donor, &donor],
        },
        OverlayStructValidityInput {
            routing: &route,
            file_order: &[0, 1],
            sources: &[&donor, &short],
        },
    ] {
        let error = AdmittedStructValidityRouting::try_new(&base, &[input], &mut zero)
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("struct validity"));
        assert_eq!(zero.len(), 0);
    }
    let input = OverlayStructValidityInput {
        routing: &route,
        file_order: &[0, 1],
        sources: &[&donor, &donor],
    };
    let error = AdmittedStructValidityRouting::try_new(&donor, &[input], &mut zero)
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("row/source counts"));
}

#[rstest]
fn resolved_ancestors_restore_simultaneously(#[values(false, true)] reverse: bool) {
    let values: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
    let child = StructArray::new(
        vec![Arc::new(ArrowField::new("value", DataType::Utf8, false))].into(),
        vec![values],
        Some(NullBuffer::from(vec![false, true])),
    );
    let root = StructArray::new(
        vec![Arc::new(ArrowField::new(
            "child",
            child.data_type().clone(),
            false,
        ))]
        .into(),
        vec![Arc::new(child.clone())],
        Some(NullBuffer::from(vec![false, true])),
    );
    let donor = validity(vec![true]);
    let coverage: RoaringBitmap = [0].into_iter().collect();
    let budget = AllocationBudget::new(4096);
    let mut work = budget.try_reserve(4096).unwrap();
    let routing = AdmittedOverlayRouting::try_new(
        &OverlayBatchSelection::Range(0..2),
        &[&coverage],
        &mut work,
    )
    .unwrap()
    .unwrap();
    let inputs = [OverlayStructValidityInput {
        routing: &routing,
        file_order: &[0],
        sources: &[&donor],
    }];
    let outer = AdmittedStructValidityRouting::try_new(&root, &inputs, &mut work)
        .unwrap()
        .unwrap();
    let inner = AdmittedStructValidityRouting::try_new(&child, &inputs, &mut work)
        .unwrap()
        .unwrap();
    let mut updates = [outer.selection(&[]), inner.selection(&[0])];
    if reverse {
        updates.reverse();
    }
    let output = try_replace_struct_validities_in(&root, &updates, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(output.null_count(), 0);
    assert_eq!(output.as_struct().column(0).null_count(), 0);
    assert_eq!(
        output.as_struct().column(0).as_struct().column(0).to_data(),
        child.column(0).to_data()
    );
}
