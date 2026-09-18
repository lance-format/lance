// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Run-length encoded holes of a [`U64Segment::Ranges`](super::U64Segment).

use std::ops::Range;

use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;

use super::bitmap::Bitmap;
use super::serde::corrupt_row_id_metadata;

/// The missing offsets of a range, as maximal runs.
///
/// Offsets are relative to the range start and fit `u32`, which bounds the span
/// of a range this encoding can describe to `u32::MAX`; a fragment never holds
/// more rows than that, and writers fall back to another encoding for wider
/// spans. Compared to a bitmap this costs 8 bytes per run rather than one bit
/// per offset, so it pays off exactly when deletions cluster, as they do after
/// compaction of fragments with deleted rows.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct HoleRuns {
    /// Length of the range the runs live in (offsets are below this).
    span: u32,
    /// First missing offset of each run, strictly increasing.
    starts: Vec<u32>,
    /// `present_before[k]` is the number of present offsets before run `k`;
    /// the final entry (index `num_runs`) is the number present in total. Run
    /// ends follow from it: the offsets missing before run `k` are
    /// `starts[k] - present_before[k]`, and a run is as long as the missing
    /// count grows across it.
    present_before: Vec<u32>,
}

impl HoleRuns {
    /// Build from run bounds. Rejects runs that are empty, out of order,
    /// overlapping, adjacent (a maximal run never touches the next) or beyond
    /// `span`.
    pub fn try_new(span: u32, starts: Vec<u32>, ends: Vec<u32>) -> Result<Self> {
        if starts.len() != ends.len() {
            return Err(corrupt_row_id_metadata(format!(
                "Ranges has {} run starts but {} run ends",
                starts.len(),
                ends.len()
            )));
        }
        let mut present_before = Vec::with_capacity(starts.len() + 1);
        let mut missing: u32 = 0;
        let mut previous_end: Option<u32> = None;
        for (i, (&start, &end)) in starts.iter().zip(&ends).enumerate() {
            if start >= end {
                return Err(corrupt_row_id_metadata(format!(
                    "Ranges run {i} is empty or reversed: {start}..{end}"
                )));
            }
            if end > span {
                return Err(corrupt_row_id_metadata(format!(
                    "Ranges run {i} ({start}..{end}) ends beyond the span {span}"
                )));
            }
            if let Some(previous_end) = previous_end
                && previous_end >= start
            {
                return Err(corrupt_row_id_metadata(format!(
                    "Ranges run {i} starts at {start}, but the previous run ends at \
                     {previous_end}: runs must be sorted, disjoint and non-adjacent"
                )));
            }
            present_before.push(start - missing);
            // Disjoint runs below `span` cannot miss more than `span` offsets.
            missing += end - start;
            previous_end = Some(end);
        }
        present_before.push(span - missing);
        Ok(Self {
            span,
            starts,
            present_before,
        })
    }

    /// The runs of cleared bits of `bitmap`, whose length must fit `u32`.
    ///
    /// Walks bytes rather than bits: on a compacted table almost every byte is
    /// all-present or all-missing, and the bit loop only runs at run edges.
    pub fn from_bitmap(bitmap: &Bitmap) -> Self {
        let span = bitmap.len();
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut open_run: Option<u32> = None;
        let mut offset: usize = 0;
        for &byte in bitmap.bytes() {
            let valid_bits = (span - offset).min(8);
            // Bits past `span` read as present so they never open a run.
            let mut present = byte | (u8::MAX.checked_shl(valid_bits as u32).unwrap_or(0));
            if present == u8::MAX {
                if let Some(start) = open_run.take() {
                    starts.push(start);
                    ends.push(offset as u32);
                }
            } else if present == 0 {
                open_run.get_or_insert(offset as u32);
            } else {
                for bit in 0..valid_bits {
                    let is_present = present & 1 != 0;
                    present >>= 1;
                    let here = (offset + bit) as u32;
                    match (is_present, open_run) {
                        (false, None) => open_run = Some(here),
                        (true, Some(start)) => {
                            starts.push(start);
                            ends.push(here);
                            open_run = None;
                        }
                        _ => {}
                    }
                }
            }
            offset += 8;
            if offset >= span {
                break;
            }
        }
        if let Some(start) = open_run {
            starts.push(start);
            ends.push(span as u32);
        }
        Self::try_new(span as u32, starts, ends).expect("runs derived from a bitmap are valid")
    }

    pub fn num_runs(&self) -> usize {
        self.starts.len()
    }

    /// Offsets present over the whole span.
    pub fn present_len(&self) -> u32 {
        self.present_before[self.starts.len()]
    }

    /// Start of run `k`, or the span end once the runs are exhausted.
    fn start_or_span(&self, k: usize) -> u32 {
        self.starts.get(k).copied().unwrap_or(self.span)
    }

    /// Offsets missing in runs `0..k`.
    fn missing_before(&self, k: usize) -> u32 {
        self.start_or_span(k) - self.present_before[k]
    }

    /// First offset after run `k`.
    fn end(&self, k: usize) -> u32 {
        self.starts[k] + (self.missing_before(k + 1) - self.missing_before(k))
    }

    /// Position of the present `offset`, or `None` when it is missing or out of range.
    pub fn position(&self, offset: u32) -> Option<u32> {
        if offset >= self.span {
            return None;
        }
        let k = self.starts.partition_point(|&start| start <= offset);
        if k > 0 && offset < self.end(k - 1) {
            return None;
        }
        Some(offset - self.missing_before(k))
    }

    /// Runs entirely before the present value at `position`.
    fn runs_before_position(&self, position: u32) -> usize {
        self.present_before[..self.starts.len()].partition_point(|&present| present <= position)
    }

    /// Offset of the present value at `position`.
    pub fn offset_at(&self, position: u32) -> Option<u32> {
        if position >= self.present_len() {
            return None;
        }
        let k = self.runs_before_position(position);
        Some(position + self.missing_before(k))
    }

    /// The present offsets, as the maximal ranges between runs.
    pub fn present_ranges(&self) -> impl DoubleEndedIterator<Item = Range<u32>> + '_ {
        (0..=self.starts.len())
            .map(move |k| {
                let start = if k == 0 { 0 } else { self.end(k - 1) };
                start..self.start_or_span(k)
            })
            .filter(|range| !range.is_empty())
    }

    /// Append `base + offset` for the present offsets at `positions`.
    ///
    /// Streaming readers materialize `_rowid` a batch at a time through this;
    /// walking the present ranges keeps that linear instead of a binary search
    /// per row.
    pub fn extend_values(&self, base: u64, positions: Range<u32>, values: &mut Vec<u64>) {
        let end = positions.end.min(self.present_len());
        let mut position = positions.start;
        let mut k = self.runs_before_position(position);
        while position < end && k <= self.starts.len() {
            // Present range between run k - 1 and run k, in offsets and positions.
            let gap_start = if k == 0 { 0 } else { self.end(k - 1) };
            let gap_end = self.start_or_span(k);
            let first_position = gap_start - self.missing_before(k);
            let gap_positions = first_position..first_position + (gap_end - gap_start);
            let take = position.max(gap_positions.start)..end.min(gap_positions.end);
            if !take.is_empty() {
                let offset = gap_start + (take.start - gap_positions.start);
                values.extend((base + offset as u64)..(base + offset as u64 + take.len() as u64));
                position = take.end;
            }
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs() -> HoleRuns {
        // span 20: present 0..3, missing 3..7, present 7..8, missing 8..15, present 15..20
        HoleRuns::try_new(20, vec![3, 8], vec![7, 15]).unwrap()
    }

    #[test]
    fn test_counts_and_ranges() {
        let runs = runs();
        assert_eq!(runs.span - runs.present_len(), 11);
        assert_eq!(runs.present_len(), 9);
        assert_eq!((0..2).map(|k| runs.end(k)).collect::<Vec<_>>(), vec![7, 15]);
        assert_eq!(
            runs.present_ranges().collect::<Vec<_>>(),
            vec![0..3, 7..8, 15..20]
        );
        assert_eq!(
            runs.present_ranges().rev().collect::<Vec<_>>(),
            vec![15..20, 7..8, 0..3]
        );
    }

    #[test]
    fn test_position_and_offset_round_trip() {
        let runs = runs();
        let present: Vec<u32> = runs.present_ranges().flatten().collect();
        assert_eq!(present, vec![0, 1, 2, 7, 15, 16, 17, 18, 19]);
        for (position, &offset) in present.iter().enumerate() {
            assert_eq!(
                runs.position(offset),
                Some(position as u32),
                "offset {offset}"
            );
            assert_eq!(
                runs.offset_at(position as u32),
                Some(offset),
                "position {position}"
            );
        }
        for missing in [3, 4, 6, 8, 14, 20, 100] {
            assert_eq!(runs.position(missing), None, "offset {missing}");
        }
        assert_eq!(runs.offset_at(9), None);
    }

    #[test]
    fn test_extend_values_crosses_runs() {
        let runs = runs();
        let mut values = Vec::new();
        runs.extend_values(100, 1..7, &mut values);
        assert_eq!(values, vec![101, 102, 107, 115, 116, 117]);
        values.clear();
        runs.extend_values(100, 4..40, &mut values);
        assert_eq!(values, vec![115, 116, 117, 118, 119]);
        values.clear();
        runs.extend_values(100, 9..12, &mut values);
        assert!(values.is_empty());
    }

    #[test]
    fn test_from_bitmap_matches_cleared_bits() {
        for (len, cleared) in [
            (20usize, vec![3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]),
            (1, vec![0]),
            (9, vec![]),
            (17, vec![0, 16]),
            (64, (8..56).collect()),
            (70, vec![7, 8, 9, 15, 16, 63, 64, 65, 66, 67, 68, 69]),
        ] {
            let mut bitmap = Bitmap::new_full(len);
            for &offset in &cleared {
                bitmap.clear(offset);
            }
            let runs = HoleRuns::from_bitmap(&bitmap);
            let missing: Vec<u32> = (0..len as u32)
                .filter(|offset| runs.position(*offset).is_none())
                .collect();
            assert_eq!(
                missing,
                cleared.iter().map(|&c| c as u32).collect::<Vec<_>>(),
                "len {len}"
            );
            assert_eq!((runs.span - runs.present_len()) as usize, cleared.len());
            // Runs are maximal: each ends strictly before the next starts.
            for k in 1..runs.starts.len() {
                assert!(runs.end(k - 1) < runs.starts[k]);
            }
        }
    }

    #[test]
    fn test_try_new_rejects_malformed_runs() {
        let cases: [(&str, Vec<u32>, Vec<u32>); 5] = [
            ("length mismatch", vec![1], vec![]),
            ("empty run", vec![3], vec![3]),
            ("beyond span", vec![3], vec![21]),
            ("overlapping", vec![3, 5], vec![7, 9]),
            ("adjacent", vec![3, 7], vec![7, 9]),
        ];
        for (name, starts, ends) in cases {
            let error = HoleRuns::try_new(20, starts, ends).unwrap_err();
            assert!(error.to_string().contains("Ranges"), "{name}: {error}");
        }
    }
}
