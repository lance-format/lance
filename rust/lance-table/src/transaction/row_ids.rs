// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stable row-id assignment for fragments that don't yet carry one.
//!
//! Extracted from `lance::dataset::transaction::Transaction::assign_row_ids` so
//! the action-based code paths (e.g. `RefreshRowVersionMetadata::apply`) and
//! the legacy `Transaction::build_manifest` can share the same implementation
//! without the latter being a hard dependency of the former.

use std::cmp::Ordering;

use crate::format::{Fragment, RowIdMeta};
use crate::rowids::{RowIdSequence, read_row_ids, write_row_ids};
use lance_core::{Error, Result};

/// Assign stable row ids to `fragments`, advancing `next_row_id` past the
/// last row id written.
///
/// Fragments that already have a complete `row_id_meta` are left alone. A
/// fragment with a partial `row_id_meta` (e.g. from a merge-insert that only
/// captured ids for the inserted rows) has the remainder filled from
/// `next_row_id`. A fragment with no `row_id_meta` gets one written from
/// `next_row_id` covering all its physical rows.
pub fn assign_row_ids(next_row_id: &mut u64, fragments: &mut [Fragment]) -> Result<()> {
    for fragment in fragments {
        let physical_rows = fragment
            .physical_rows
            .ok_or_else(|| Error::internal("Fragment does not have physical rows"))?
            as u64;

        if fragment.row_id_meta.is_some() {
            // we may meet merge insert case, it only has partial row ids.
            // so here, we need to check if the row ids match the physical rows
            // if yes, continue
            // if not, fill the remaining row ids to the physical rows, then update row_id_meta

            // Check if existing row IDs match the physical rows count
            let existing_row_count = match &fragment.row_id_meta {
                Some(RowIdMeta::Inline(data)) => {
                    // Parse the serialized row ID sequence to get the count
                    let sequence = read_row_ids(data)?;
                    sequence.len() as u64
                }
                _ => 0,
            };

            match existing_row_count.cmp(&physical_rows) {
                Ordering::Equal => {
                    // Row IDs already match physical rows, continue to next fragment
                    continue;
                }
                Ordering::Less => {
                    // Partial row IDs - need to fill the remaining ones
                    let remaining_rows = physical_rows - existing_row_count;
                    let new_row_ids = *next_row_id..(*next_row_id + remaining_rows);

                    // Merge existing and new row IDs
                    let combined_sequence = match &fragment.row_id_meta {
                        Some(RowIdMeta::Inline(data)) => read_row_ids(data)?,
                        _ => {
                            return Err(Error::internal(
                                "Failed to deserialize existing row ID sequence",
                            ));
                        }
                    };

                    let mut row_ids: Vec<u64> = combined_sequence.iter().collect();
                    for row_id in new_row_ids {
                        row_ids.push(row_id);
                    }
                    let combined_sequence = RowIdSequence::from(row_ids.as_slice());

                    let serialized = write_row_ids(&combined_sequence);
                    fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
                    *next_row_id += remaining_rows;
                }
                Ordering::Greater => {
                    // More row IDs than physical rows - this shouldn't happen
                    return Err(Error::internal(format!(
                        "Fragment has more row IDs ({}) than physical rows ({})",
                        existing_row_count, physical_rows
                    )));
                }
            }
        } else {
            let row_ids = *next_row_id..(*next_row_id + physical_rows);
            let sequence = RowIdSequence::from(row_ids);
            // TODO: write to a separate file if large. Possibly share a file with other fragments.
            let serialized = write_row_ids(&sequence);
            fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
            *next_row_id += physical_rows;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fragment(id: u64, physical_rows: usize) -> Fragment {
        let mut fragment = Fragment::new(id);
        fragment.physical_rows = Some(physical_rows);
        fragment
    }

    #[test]
    fn test_assign_row_ids_new_fragment() {
        let mut next_row_id = 0;
        let mut fragments = vec![make_fragment(1, 5)];

        assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        assert_eq!(next_row_id, 5);
        assert!(fragments[0].row_id_meta.is_some());
        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 5);
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_assign_row_ids_existing_complete() {
        let mut next_row_id = 10;
        let existing_ids: Vec<u64> = vec![0, 1, 2, 3, 4];
        let existing_sequence = RowIdSequence::from(existing_ids.as_slice());
        let serialized = write_row_ids(&existing_sequence);

        let mut fragment = make_fragment(1, 5);
        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
        let mut fragments = vec![fragment];

        assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        // next_row_id should not change since row IDs already match
        assert_eq!(next_row_id, 10);
        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_assign_row_ids_partial_existing() {
        let mut next_row_id = 100;
        let existing_ids: Vec<u64> = vec![10, 11, 12];
        let existing_sequence = RowIdSequence::from(existing_ids.as_slice());
        let serialized = write_row_ids(&existing_sequence);

        let mut fragment = make_fragment(1, 5);
        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
        let mut fragments = vec![fragment];

        assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        // Should add 2 more row IDs (100, 101) to make 5 total
        assert_eq!(next_row_id, 102);
        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 5);
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, vec![10, 11, 12, 100, 101]);
        }
    }

    #[test]
    fn test_assign_row_ids_excess_row_ids() {
        let mut next_row_id = 100;
        let existing_ids: Vec<u64> = vec![10, 11, 12, 13, 14, 15];
        let existing_sequence = RowIdSequence::from(existing_ids.as_slice());
        let serialized = write_row_ids(&existing_sequence);

        let mut fragment = make_fragment(1, 5);
        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
        let mut fragments = vec![fragment];

        let result = assign_row_ids(&mut next_row_id, &mut fragments);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("more row IDs"));
    }

    #[test]
    fn test_assign_row_ids_multiple_fragments() {
        let mut next_row_id = 0;

        // First fragment: no existing row IDs (new fragment)
        let fragment1 = make_fragment(1, 3);

        // Second fragment: partial existing row IDs
        let existing_ids: Vec<u64> = vec![100, 101];
        let existing_sequence = RowIdSequence::from(existing_ids.as_slice());
        let serialized = write_row_ids(&existing_sequence);
        let mut fragment2 = make_fragment(2, 4);
        fragment2.row_id_meta = Some(RowIdMeta::Inline(serialized));

        // Third fragment: complete existing row IDs
        let complete_ids: Vec<u64> = vec![200, 201];
        let complete_sequence = RowIdSequence::from(complete_ids.as_slice());
        let complete_serialized = write_row_ids(&complete_sequence);
        let mut fragment3 = make_fragment(3, 2);
        fragment3.row_id_meta = Some(RowIdMeta::Inline(complete_serialized));

        let mut fragments = vec![fragment1, fragment2, fragment3];

        assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        // First fragment: assigns 0, 1, 2 → next_row_id becomes 3
        // Second fragment: needs 2 more (100, 101 + 3, 4) → next_row_id becomes 5
        // Third fragment: already complete → no change
        assert_eq!(next_row_id, 5);

        // Verify first fragment
        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, vec![0, 1, 2]);
        }

        // Verify second fragment
        if let Some(RowIdMeta::Inline(data)) = &fragments[1].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, vec![100, 101, 3, 4]);
        }

        // Verify third fragment unchanged
        if let Some(RowIdMeta::Inline(data)) = &fragments[2].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, vec![200, 201]);
        }
    }

    #[test]
    fn test_assign_row_ids_missing_physical_rows() {
        let mut next_row_id = 0;
        let mut fragments = vec![Fragment::new(1)];

        let result = assign_row_ids(&mut next_row_id, &mut fragments);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("physical rows"));
    }
}
