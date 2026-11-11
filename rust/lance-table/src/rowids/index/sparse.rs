// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{EncodedU64Array, Error, FragmentRowIdIndex, Result, RowAddress, U64Segment};

fn position(segment: &U64Segment, id: u64) -> Option<usize> {
    // Keep the array representation dispatch outside the scan. The generic
    // encoded-array iterator uses dynamic dispatch for each value.
    match segment {
        U64Segment::Array(EncodedU64Array::U16 { base, offsets }) => {
            let relative = u16::try_from(id.checked_sub(*base)?).ok()?;
            offsets.iter().position(|&value| value == relative)
        }
        U64Segment::Array(EncodedU64Array::U32 { base, offsets }) => {
            let relative = u32::try_from(id.checked_sub(*base)?).ok()?;
            offsets.iter().position(|&value| value == relative)
        }
        U64Segment::Array(EncodedU64Array::U64(values)) => {
            values.iter().position(|&value| value == id)
        }
        _ => segment.position(id),
    }
}

impl FragmentRowIdIndex {
    /// Resolve a small set of row IDs directly, without constructing an index.
    ///
    /// Returns one address per requested ID, preserving order and duplicates;
    /// missing and deleted IDs return `None`. Encoded ranges and sorted arrays
    /// use their native lookup. An unsorted segment is scanned once per distinct
    /// requested ID, so a prepared [`super::RowIdIndex`] is preferable for large
    /// batches or repeated queries. Temporary memory is proportional to the
    /// request, not the fragment's row count.
    ///
    /// This only searches this fragment. A dataset caller must visit every
    /// candidate fragment and reject IDs live in more than one fragment. Finding
    /// a match does not establish that other fragments can safely be skipped.
    /// Returns an error for multiple live segment matches or physical offsets
    /// that cannot be represented by a [`RowAddress`].
    ///
    /// ```
    /// # use lance_table::rowids::FragmentRowIdIndex;
    /// # fn example(fragment: &FragmentRowIdIndex) -> lance_core::Result<()> {
    /// let addresses = fragment.get_many(&[42, 42, 100])?;
    /// assert_eq!(addresses[0], addresses[1]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_many(&self, row_ids: &[u64]) -> Result<Vec<Option<RowAddress>>> {
        let mut queries: Vec<_> = row_ids.iter().copied().zip(0..row_ids.len()).collect();
        queries.sort_unstable_by_key(|&(id, _)| id);
        let mut output = vec![None; row_ids.len()];
        let mut previous = None;
        let mut address = None;
        for (id, ordinal) in queries {
            if previous != Some(id) {
                address = None;
                let mut start_offset = 0_u64;
                for segment in &self.row_id_sequence.0 {
                    if let Some(position) = position(segment, id) {
                        let offset =
                            start_offset.checked_add(position as u64).ok_or_else(|| {
                                Error::invalid_input(format!(
                                    "fragment {} row offset overflows u64 for row ID {id}",
                                    self.fragment_id
                                ))
                            })?;
                        let offset = u32::try_from(offset).map_err(|_| {
                            Error::invalid_input(format!(
                                "fragment {} row offset {offset} exceeds u32 for row ID {id}",
                                self.fragment_id
                            ))
                        })?;
                        if !self.deletion_vector.contains(offset) {
                            if address.is_some() {
                                return Err(Error::internal(format!(
                                    "row id index corrupt: stable row id {id} is live in \
                                     multiple segments of fragment {}",
                                    self.fragment_id
                                )));
                            }
                            address = Some(RowAddress::new_from_parts(self.fragment_id, offset));
                        }
                    }
                    start_offset =
                        start_offset
                            .checked_add(segment.len() as u64)
                            .ok_or_else(|| {
                                Error::invalid_input(format!(
                                    "fragment {} physical row count overflows u64",
                                    self.fragment_id
                                ))
                            })?;
                }
                previous = Some(id);
            }
            output[ordinal] = address;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rowids::{RowIdSequence, U64Segment};
    use lance_core::utils::deletion::DeletionVector;
    use std::sync::Arc;

    #[rstest::rstest]
    #[case::range(U64Segment::Range(10..20))]
    #[case::holes(U64Segment::from_slice(&[10, 12, 13, 15, 16, 19]))]
    #[case::sorted(U64Segment::SortedArray(vec![10, 12, 19, 30].into()))]
    #[case::shuffled(U64Segment::Array(vec![15, 12, 19, 10, 13, 16].into()))]
    #[case::wide(U64Segment::Array(vec![u64::MAX, 0, 1 << 40, 42].into()))]
    #[case::u32(U64Segment::Array(vec![42, 1 << 20, 12, 1 << 21].into()))]
    fn direct_matches_physical_slots(#[case] segment: U64Segment) {
        let ids: Vec<_> = segment.iter().collect();
        let source = FragmentRowIdIndex {
            fragment_id: 7,
            row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(5..7), segment])),
            deletion_vector: Arc::new(DeletionVector::Bitmap([1, 3].into_iter().collect())),
        };
        let physical: Vec<_> = [5, 6].into_iter().chain(ids).collect();
        let queries: Vec<_> = physical
            .iter()
            .copied()
            .rev()
            .chain([6, 42, 5, 5, u64::MAX])
            .collect();
        let expected: Vec<_> = queries
            .iter()
            .map(|id| {
                physical
                    .iter()
                    .position(|v| v == id)
                    .filter(|&offset| ![1, 3].contains(&offset))
                    .map(|offset| RowAddress::new_from_parts(7, offset as u32))
            })
            .collect();
        assert_eq!(source.get_many(&queries).unwrap(), expected);
        assert!(source.get_many(&[]).unwrap().is_empty());
    }

    #[test]
    fn direct_rejects_multiple_live_segments() {
        let mut source = FragmentRowIdIndex {
            fragment_id: 9,
            row_id_sequence: Arc::new(RowIdSequence(vec![
                U64Segment::Range(0..3),
                U64Segment::Range(2..4),
            ])),
            deletion_vector: Arc::new(DeletionVector::default()),
        };
        let error = source.get_many(&[2, 2]).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(error.to_string().contains("row id 2"));
        source.deletion_vector = Arc::new(DeletionVector::Bitmap([2].into_iter().collect()));
        assert_eq!(
            source.get_many(&[2, 2]).unwrap(),
            vec![Some(RowAddress::new_from_parts(9, 3)); 2]
        );
    }

    #[test]
    fn direct_rejects_unrepresentable_address() {
        let source = FragmentRowIdIndex {
            fragment_id: 9,
            row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(
                0..u32::MAX as u64 + 2,
            )])),
            deletion_vector: Arc::new(DeletionVector::default()),
        };
        let error = source.get_many(&[u32::MAX as u64 + 1]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("row offset 4294967296"));
    }
}
