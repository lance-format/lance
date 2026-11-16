// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep read selections compact until an individual overlay batch is consumed.

use std::{ops::Range, sync::Arc};

use arrow_array::UInt32Array;
use lance_core::{Error, Result};
use lance_io::ReadBatchParams;
use roaring::RoaringBitmap;

enum Selection {
    Range(Range<u32>),
    Ranges {
        ranges: Arc<[Range<u64>]>,
        // One prefix count per range, not per selected row. Locating a batch
        // must not rescan all preceding ranges for every output batch.
        ends: Vec<usize>,
    },
    // Arbitrary takes already supply explicit offsets. Preserve their bitmap
    // pruning cost without building one for a potentially huge scan.
    Indices(UInt32Array, RoaringBitmap),
}

/// A batch keeps contiguous row identity without an offsets allocation.
pub enum OverlayBatchSelection {
    /// A half-open range of physical row offsets within one fragment.
    Range(Range<u32>),
    /// Explicit physical row offsets, preserving caller order and duplicates.
    /// Admitted routing rejects null entries.
    Indices(UInt32Array),
}

pub struct OverlayReadSelection {
    selection: Selection,
    num_rows: usize,
}

impl OverlayReadSelection {
    pub(crate) fn new(params: &ReadBatchParams, total: u32) -> Result<Self> {
        let range_len = |start: u64, end: u64| {
            if start > end || end > u64::from(total) {
                return Err(Error::invalid_input(format!(
                    "overlay selection {start}..{end} exceeds {total} physical rows"
                )));
            }
            Ok((end - start) as usize)
        };
        let range = |start: u64, end: u64| -> Result<Self> {
            let num_rows = range_len(start, end)?;
            Ok(Self {
                selection: Selection::Range(start as u32..end as u32),
                num_rows,
            })
        };
        match params {
            ReadBatchParams::RangeFull => range(0, u64::from(total)),
            ReadBatchParams::RangeFrom(r) => range(r.start as u64, u64::from(total)),
            ReadBatchParams::RangeTo(r) => range(0, r.end as u64),
            ReadBatchParams::Range(r) => range(r.start as u64, r.end as u64),
            ReadBatchParams::Ranges(ranges) => {
                let mut ends = Vec::with_capacity(ranges.len());
                let mut num_rows = 0_usize;
                for r in ranges.iter() {
                    num_rows = num_rows
                        .checked_add(range_len(r.start, r.end)?)
                        .ok_or_else(|| {
                            Error::invalid_input("overlay selection row count overflow")
                        })?;
                    ends.push(num_rows);
                }
                Ok(Self {
                    selection: Selection::Ranges {
                        ranges: ranges.clone(),
                        ends,
                    },
                    num_rows,
                })
            }
            ReadBatchParams::Indices(indices) => {
                if indices.values().iter().any(|offset| *offset >= total) {
                    return Err(Error::invalid_input(format!(
                        "overlay selection index exceeds {total} physical rows"
                    )));
                }
                let bitmap = indices.values().iter().copied().collect();
                Ok(Self {
                    selection: Selection::Indices(indices.clone(), bitmap),
                    num_rows: indices.len(),
                })
            }
        }
    }

    pub(super) fn intersects(&self, coverage: &RoaringBitmap) -> bool {
        match &self.selection {
            Selection::Range(range) => coverage.range_cardinality(range.clone()) > 0,
            Selection::Ranges { ranges, .. } => ranges
                .iter()
                .any(|range| coverage.range_cardinality(range.start as u32..range.end as u32) > 0),
            Selection::Indices(_, indices) => !coverage.is_disjoint(indices),
        }
    }

    /// Slice this batch's physical rows, preserving order and repeats.
    /// Contiguous reads stay a range through routing; only disjoint ranges need
    /// an offsets buffer, and only for the batch being consumed.
    pub(crate) fn batch(&self, start: usize, len: usize) -> Result<OverlayBatchSelection> {
        if start.checked_add(len).is_none_or(|end| end > self.num_rows) {
            return Err(Error::invalid_input(format!(
                "overlay batch {start} + {len} exceeds {} selected rows",
                self.num_rows
            )));
        }
        match &self.selection {
            Selection::Range(range) => {
                let begin = range.start + start as u32;
                Ok(OverlayBatchSelection::Range(begin..begin + len as u32))
            }
            Selection::Indices(indices, _) => {
                Ok(OverlayBatchSelection::Indices(indices.slice(start, len)))
            }
            Selection::Ranges { ranges, ends } => {
                // ReadBatchParams::slice reserves descriptors for every source range.
                // Build only the consumed batch, including when it crosses ranges.
                let mut offsets = Vec::with_capacity(len);
                let first = ends.partition_point(|end| *end <= start);
                let preceding_rows = if first == 0 { 0 } else { ends[first - 1] };
                let mut skip = (start - preceding_rows) as u64;
                for range in &ranges[first..] {
                    if offsets.len() == len {
                        break;
                    }
                    let count = range.end - range.start;
                    if skip >= count {
                        skip -= count;
                        continue;
                    }
                    let begin = range.start + skip;
                    let count = (range.end - begin).min((len - offsets.len()) as u64);
                    offsets.extend(begin as u32..(begin + count) as u32);
                    skip = 0;
                }
                Ok(OverlayBatchSelection::Indices(UInt32Array::from(offsets)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn offsets(selection: OverlayBatchSelection) -> Vec<u32> {
        match selection {
            OverlayBatchSelection::Range(range) => range.collect(),
            OverlayBatchSelection::Indices(indices) => indices.values().to_vec(),
        }
    }

    #[rstest]
    #[case::full(ReadBatchParams::RangeFull, vec![0, 1, 2, 3, 4, 5, 6, 7])]
    #[case::range(ReadBatchParams::Range(2..6), vec![2, 3, 4, 5])]
    #[case::from(ReadBatchParams::RangeFrom(5..), vec![5, 6, 7])]
    #[case::to(ReadBatchParams::RangeTo(..3), vec![0, 1, 2])]
    #[case::ranges(ReadBatchParams::Ranges(vec![5..8, 0..0, 1..4, 2..3].into()), vec![5, 6, 7, 1, 2, 3, 2])]
    #[case::indices(ReadBatchParams::Indices(UInt32Array::from(vec![7, 1, 7, 0])), vec![7, 1, 7, 0])]
    #[case::empty(ReadBatchParams::Ranges(vec![3..3, 8..8].into()), vec![])]
    fn batch_slices_and_pruning_match_selection(
        #[case] params: ReadBatchParams,
        #[case] expected: Vec<u32>,
    ) {
        let selection = OverlayReadSelection::new(&params, 8).unwrap();
        for start in 0..=expected.len() {
            for len in 0..=expected.len() - start {
                assert_eq!(
                    offsets(selection.batch(start, len).unwrap()).as_slice(),
                    &expected[start..start + len]
                );
            }
        }
        for offsets in [vec![], vec![0], vec![2, 6], vec![7, 8]] {
            let coverage = offsets.iter().copied().collect();
            assert_eq!(
                selection.intersects(&coverage),
                expected.iter().any(|offset| offsets.contains(offset))
            );
        }
        let error = selection.batch(expected.len(), 1).err().unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("selected rows"));
    }

    #[test]
    fn full_fragment_does_not_expand_before_batch_consumption() {
        // Expanding this selection up front would require almost 16 GiB. Both
        // selection and batch descriptions must remain compact near the end.
        let selection = OverlayReadSelection::new(&ReadBatchParams::RangeFull, u32::MAX).unwrap();
        // Even a declared whole-fragment batch retains a compact range. Do not
        // expand it in this test: the physical row identity needs no allocation.
        assert!(matches!(
            selection.batch(0, u32::MAX as usize).unwrap(),
            OverlayBatchSelection::Range(range) if range == (0..u32::MAX)
        ));
        let coverage = [u32::MAX - 1].into_iter().collect();
        assert!(selection.intersects(&coverage));
        assert!(!selection.intersects(&[u32::MAX].into_iter().collect()));
        assert_eq!(
            offsets(selection.batch(u32::MAX as usize - 2, 2).unwrap()).as_slice(),
            &[u32::MAX - 2, u32::MAX - 1]
        );
        assert_eq!(
            offsets(selection.batch(u32::MAX as usize, 0).unwrap()).len(),
            0
        );
    }

    #[test]
    fn large_ranges_preserve_a_batch_crossing_the_boundary() {
        let selection = OverlayReadSelection::new(
            &ReadBatchParams::Ranges(vec![100..u64::from(u32::MAX), 0..2].into()),
            u32::MAX,
        )
        .unwrap();
        assert_eq!(
            offsets(selection.batch(u32::MAX as usize - 101, 3).unwrap()).as_slice(),
            &[u32::MAX - 1, 0, 1]
        );
        assert!(!selection.intersects(&[99].into_iter().collect()));
    }

    #[rstest]
    #[case::reversed(ReadBatchParams::Range(std::ops::Range { start: 4, end: 2 }))]
    #[case::past_end(ReadBatchParams::RangeTo(..9))]
    #[case::bad_range(ReadBatchParams::Ranges(vec![0..1, 4..9].into()))]
    #[case::bad_index(ReadBatchParams::Indices(UInt32Array::from(vec![8])))]
    fn invalid_selection_fails_before_materialization(#[case] params: ReadBatchParams) {
        let error = OverlayReadSelection::new(&params, 8).err().unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("physical rows"));
    }
}
