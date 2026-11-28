// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::UInt64Array;
use lance_core::utils::{address::RowAddress, deletion::DeletionVector};
use lance_core::{Error, Result};

/// The logical starting offset and physical deletions of one table fragment.
#[derive(Debug)]
pub struct FragmentRowOffset {
    /// Number of live rows in all preceding fragments in table order.
    pub row_offset: u64,
    /// Deleted physical row offsets within this fragment, if it has deletions.
    pub deletion_vector: Option<Arc<DeletionVector>>,
}

/// Translate non-null addresses of live rows into zero-based table offsets.
///
/// The fragment offsets must come from the same snapshot as the addresses.
/// The result preserves input order and duplicates. An address in a fragment
/// missing from the supplied offsets returns an error.
///
/// ```
/// use std::collections::HashMap;
/// use arrow_array::UInt64Array;
/// use lance_core::utils::address::RowAddress;
/// use lance_table::rowids::{FragmentRowOffset, compute_row_offset_array};
///
/// let addresses = UInt64Array::from(vec![u64::from(RowAddress::new_from_parts(7, 2))]);
/// let fragments = HashMap::from([(
///     7,
///     FragmentRowOffset { row_offset: 10, deletion_vector: None },
/// )]);
/// assert_eq!(compute_row_offset_array(&addresses, &fragments)?.value(0), 12);
/// # Ok::<(), lance_core::Error>(())
/// ```
pub fn compute_row_offset_array(
    row_addrs: &UInt64Array,
    frag_id_to_offset: &HashMap<u32, FragmentRowOffset>,
) -> Result<UInt64Array> {
    let row_addr_values = row_addrs.values();
    let mut row_offsets = vec![0; row_addr_values.len()];
    // The deletion iterator moves only forward. Visit addresses in order and
    // scatter the offsets back into the input order.
    let mut sorted_row_indices = (0..row_addr_values.len()).collect::<Vec<_>>();
    sorted_row_indices.sort_unstable_by_key(|index| row_addr_values[*index]);

    let mut last_frag_id = None;
    let mut frag_offset = 0;
    let mut deleted_before_row = 0_u64;
    let mut deletions = None;

    for row_index in sorted_row_indices {
        let address = RowAddress::new_from_u64(row_addr_values[row_index]);
        let frag_id = address.fragment_id();
        if last_frag_id != Some(frag_id) {
            let frag_info = frag_id_to_offset.get(&frag_id).ok_or_else(|| {
                Error::internal(format!(
                    "A row address referred to a fragment {frag_id} that wasn't in the frag_id_to_offset map"
                ))
            })?;
            last_frag_id = Some(frag_id);
            frag_offset = frag_info.row_offset;
            deleted_before_row = 0;
            deletions = frag_info
                .deletion_vector
                .as_ref()
                .map(|vector| vector.to_sorted_iter().peekable());
        }

        let row_offset = address.row_offset();
        if let Some(deletions) = &mut deletions {
            while deletions.next_if(|&deleted| deleted < row_offset).is_some() {
                deleted_before_row += 1;
            }
        }
        row_offsets[row_index] = frag_offset + u64::from(row_offset) - deleted_before_row;
    }

    Ok(UInt64Array::from(row_offsets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_offsets_preserve_unsorted_addresses_and_duplicates() {
        let address =
            |frag_id, row_offset| u64::from(RowAddress::new_from_parts(frag_id, row_offset));
        let row_addrs = UInt64Array::from(vec![
            address(9, 3),
            address(2, 100),
            address(2, 50),
            address(2, 100),
        ]);
        let fragments = HashMap::from([
            (
                2,
                FragmentRowOffset {
                    row_offset: 1_000,
                    deletion_vector: Some(Arc::new(DeletionVector::from_iter([10, 60]))),
                },
            ),
            (
                9,
                FragmentRowOffset {
                    row_offset: 1_150,
                    deletion_vector: None,
                },
            ),
        ]);

        let offsets = compute_row_offset_array(&row_addrs, &fragments).unwrap();
        assert_eq!(offsets.values(), &[1_153, 1_098, 1_049, 1_098]);
    }

    #[test]
    fn row_offsets_report_missing_fragments() {
        let addresses = UInt64Array::from(vec![u64::from(RowAddress::new_from_parts(7, 0))]);
        let error = compute_row_offset_array(&addresses, &HashMap::new()).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(error.to_string().contains("fragment 7"));
    }
}
