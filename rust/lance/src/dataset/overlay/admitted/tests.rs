// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{StringArray, UInt32Array};
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;

mod file;

#[rstest]
#[case::empty(vec![])]
#[case::contiguous((0..8).collect())]
#[case::duplicates(vec![3, 0, 3, 2])]
#[case::containers(vec![65536, 65535, 65536, 65534])]
#[case::end(vec![u32::MAX, u32::MAX - 1, u32::MAX])]
fn admitted_routing_matches_existing_newest_first(#[case] rows: Vec<u32>) {
    let newer: RoaringBitmap = [1, 3, 65536, u32::MAX].into_iter().collect();
    let older: RoaringBitmap = [0, 3, 4, 65535, u32::MAX].into_iter().collect();
    let coverages = [&newer, &older];
    let expected = route_overlays(&rows, &coverages);
    let budget = AllocationBudget::new(4096);
    let mut workspace = budget.try_reserve(4096).unwrap();
    let selection = OverlayBatchSelection::Indices(UInt32Array::from(rows));
    let routing = AdmittedOverlayRouting::try_new(&selection, &coverages, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(routing.indices(), expected.indices);
    for (index, offsets) in expected.offsets_in_overlay.iter().enumerate() {
        assert_eq!(routing.fetch_offsets(index).unwrap().as_ref(), offsets);
    }
    assert_eq!(routing.any_overlay, expected.any_overlay);
    drop((routing, workspace));
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn duplicate_compaction_keeps_complete_owner_charge() {
    let newer: RoaringBitmap = [1, 3].into_iter().collect();
    let older: RoaringBitmap = [0, 3].into_iter().collect();
    let selection = OverlayBatchSelection::Indices(UInt32Array::from(vec![3, 0, 3, 2]));
    let bytes = 4 * size_of::<(usize, usize)>() + 3 * size_of::<u32>();
    let budget = AllocationBudget::new(bytes);
    let mut short = budget.try_reserve(bytes - 1).unwrap();
    assert!(
        AdmittedOverlayRouting::try_new(&selection, &[&newer, &older], &mut short)
            .unwrap()
            .is_none()
    );
    assert_eq!(short.len(), bytes - 1);
    drop(short);
    let mut workspace = budget.try_reserve(bytes).unwrap();
    let routing = AdmittedOverlayRouting::try_new(&selection, &[&newer, &older], &mut workspace)
        .unwrap()
        .unwrap();
    assert!(workspace.is_empty());
    assert_eq!(routing.indices(), &[(1, 0), (2, 0), (1, 0), (0, 3)]);
    let first = routing.fetch_offsets(0).unwrap();
    let second = routing.fetch_offsets(1).unwrap();
    assert_eq!(first.as_ref(), &[1]);
    assert_eq!(second.as_ref(), &[0]);
    drop((routing, workspace));
    assert_eq!(budget.reserved(), bytes);
    drop(first);
    assert_eq!(budget.reserved(), bytes);
    assert!(budget.try_reserve(1).is_none());
    drop(second);
    assert_eq!(budget.reserved(), 0);
    assert!(budget.try_reserve(bytes).is_some());
}

#[rstest]
#[case::descending(OverlayBatchSelection::Range(Range { start: 4, end: 3 }), "descending")]
#[case::null(OverlayBatchSelection::Indices(UInt32Array::from(vec![None])), "contains nulls")]
fn invalid_selection_precedes_capacity_refusal(
    #[case] selection: OverlayBatchSelection,
    #[case] message: &str,
) {
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let error = AdmittedOverlayRouting::try_new(&selection, &[], &mut workspace)
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message));
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn admitted_interleave_preserves_null_wins_and_duplicates() {
    let newer: RoaringBitmap = [1, 3].into_iter().collect();
    let older: RoaringBitmap = [0, 3].into_iter().collect();
    let selection = OverlayBatchSelection::Indices(UInt32Array::from(vec![3, 0, 3, 2]));
    let budget = AllocationBudget::new(4096);
    let mut workspace = budget.try_reserve(4096).unwrap();
    let routing = AdmittedOverlayRouting::try_new(&selection, &[&newer, &older], &mut workspace)
        .unwrap()
        .unwrap();
    let base: ArrayRef = Arc::new(StringArray::from(vec!["base3", "base0", "base3", "base2"]));
    let fetched: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![None::<&str>])),
        Arc::new(StringArray::from(vec!["old0"])),
    ];
    let output = routing
        .try_interleave_in(&base, &fetched, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(
        output.to_data(),
        StringArray::from(vec![None, Some("old0"), None, Some("base2")]).to_data()
    );
    drop((routing, output, workspace));
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn uncovered_range_reuses_base_and_empty_fetch_storage() {
    let coverage: RoaringBitmap = [100].into_iter().collect();
    let budget = AllocationBudget::new(128);
    let mut workspace = budget.try_reserve(128).unwrap();
    let routing = AdmittedOverlayRouting::try_new(
        &OverlayBatchSelection::Range(0..2),
        &[&coverage],
        &mut workspace,
    )
    .unwrap()
    .unwrap();
    let base: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
    let fetched: Vec<ArrayRef> = vec![Arc::new(StringArray::from(Vec::<&str>::new()))];
    let output = routing
        .try_interleave_in(&base, &fetched, &mut workspace)
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&base, &output));
    assert!(routing.fetch_offsets(0).unwrap().is_empty());
    let error = routing
        .try_interleave_in(&base, &[], &mut workspace)
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("lengths do not match"));
    drop((routing, workspace));
    assert_eq!(budget.reserved(), 0);
}
