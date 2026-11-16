// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::UInt32Array;
use arrow_buffer::{Buffer, ScalarBuffer};
use lance_core::{
    Error, Result,
    utils::{allocation::Reservation, deletion::DeletionVector},
};

/// Filter a validated physical selection without allocating a predicate or
/// copying payload. The caller retains the immutable deletion snapshot. An
/// unchanged selection shares its original backing; only a reduced selection
/// allocates offsets, from the operation's existing reservation.
pub(super) fn try_filter_in(
    indices: &UInt32Array,
    deleted: Option<&DeletionVector>,
    workspace: &mut Reservation,
) -> Result<Option<UInt32Array>> {
    let Some(deleted) = deleted.filter(|deleted| !deleted.is_empty()) else {
        return Ok(Some(indices.clone()));
    };
    let survivors = || {
        indices
            .values()
            .iter()
            .copied()
            .filter(|row| !deleted.contains(*row))
    };
    let count = survivors().count();
    if count == indices.len() {
        return Ok(Some(indices.clone()));
    }
    let bytes = count
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| Error::invalid_input("surviving physical offsets exceed address space"))?;
    let Some(mut storage) = workspace.try_allocate(bytes)? else {
        return Ok(None);
    };
    for (slot, row) in storage
        .as_mut()
        .as_chunks_mut::<{ size_of::<u32>() }>()
        .0
        .iter_mut()
        .zip(survivors())
    {
        slot.copy_from_slice(&row.to_ne_bytes());
    }
    Ok(Some(UInt32Array::new(
        ScalarBuffer::new(Buffer::from(storage.freeze()), 0, count),
        None,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use lance_core::utils::allocation::AllocationBudget;
    use rstest::rstest;

    #[rstest]
    fn filtered_offsets_keep_exact_credit_through_final_alias(#[values(false, true)] bitmap: bool) {
        let indices = UInt32Array::from(vec![1, 2, 65535, 65536, u32::MAX]);
        let deleted = if bitmap {
            DeletionVector::Bitmap([1, 65535, u32::MAX].into_iter().collect())
        } else {
            DeletionVector::Set([1, 65535, u32::MAX].into_iter().collect())
        };
        let budget = AllocationBudget::new(8);
        let mut work = budget.try_reserve(7).unwrap();
        assert!(
            try_filter_in(&indices, Some(&deleted), &mut work)
                .unwrap()
                .is_none()
        );
        assert_eq!(work.len(), 7);
        drop(work);
        let mut work = budget.try_reserve(8).unwrap();
        let output = try_filter_in(&indices, Some(&deleted), &mut work)
            .unwrap()
            .unwrap();
        assert_eq!(output.values().as_ref(), &[2, 65536]);
        assert_eq!(work.len(), 0);
        let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
        drop((output, work));
        assert_eq!(budget.reserved(), 8);
        assert!(budget.try_reserve(8).is_none());
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }

    #[rstest]
    #[case::absent(None)]
    #[case::empty(Some(DeletionVector::NoDeletions))]
    #[case::unselected(Some(DeletionVector::from_iter([0, 3])))]
    fn surviving_selection_shares_sliced_storage_without_credit(
        #[case] deleted: Option<DeletionVector>,
    ) {
        let indices = UInt32Array::from(vec![0, 1, 2, 3]).slice(1, 2);
        let budget = AllocationBudget::new(0);
        let mut work = budget.try_reserve(0).unwrap();
        let output = try_filter_in(&indices, deleted.as_ref(), &mut work)
            .unwrap()
            .unwrap();
        assert_eq!(output, indices);
        assert_eq!(output.values().as_ptr(), indices.values().as_ptr());
        assert_eq!(budget.reserved(), 0);
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::all_deleted(vec![1, 3])]
    fn empty_survivors_do_not_require_offset_storage(#[case] selected: Vec<u32>) {
        let budget = AllocationBudget::new(0);
        let mut work = budget.try_reserve(0).unwrap();
        let output = try_filter_in(
            &UInt32Array::from(selected),
            Some(&DeletionVector::from_iter([1, 3])),
            &mut work,
        )
        .unwrap()
        .unwrap();
        assert!(output.is_empty());
        assert_eq!(work.len(), 0);
        assert_eq!(budget.reserved(), 0);
    }
}
