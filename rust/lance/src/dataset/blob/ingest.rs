// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Range merge and bounded fetch for external blob ingest.
//! Placement of each payload stays with the dataset writer.

use bytes::Bytes;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::{self, Stream};
use lance_core::{Error, Result};
use lance_io::traits::Reader;

// Maximum span of one coalesced read. Larger slices stream separately.
pub(super) const INGEST_COALESCE_BUDGET: u64 = 8 * 1024 * 1024;

// Maximum number of coalesced fetches buffered at once.
pub(super) const INGEST_FETCH_PARALLELISM: usize = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ExternalSlice {
    pub row: usize,
    pub start: u64,
    pub end: u64,
}

pub(super) struct PlannedWindow {
    pub start: u64,
    pub end: u64,
    pub slices: Vec<ExternalSlice>,
}

pub(super) struct FetchWindow {
    pub start: u64,
    pub end: u64,
    pub slices: Vec<ExternalSlice>,
    pub open_reader: BoxFuture<'static, Result<Box<dyn Reader>>>,
}

pub(super) struct FetchedWindow {
    pub span_start: u64,
    pub bytes: Bytes,
    pub slices: Vec<ExternalSlice>,
}

// Plan bounded windows so ingest can write each result without collecting all reads.
pub(super) fn plan_coalesced_windows(
    mut slices: Vec<ExternalSlice>,
    hole: u64,
    byte_budget: u64,
) -> Vec<PlannedWindow> {
    if slices.is_empty() {
        return Vec::new();
    }
    slices.sort_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then(left.end.cmp(&right.end))
            .then(left.row.cmp(&right.row))
    });

    let mut windows = Vec::new();
    let mut current = Vec::new();
    let mut window_start = 0u64;
    let mut window_end = 0u64;
    for slice in slices {
        if current.is_empty() {
            window_start = slice.start;
            window_end = slice.end;
            current.push(slice);
            continue;
        }
        let merged_end = window_end.max(slice.end);
        let close = slice.start <= window_end.saturating_add(hole);
        let within_budget = merged_end.saturating_sub(window_start) <= byte_budget;
        if close && within_budget {
            window_end = merged_end;
            current.push(slice);
        } else {
            windows.push(PlannedWindow {
                start: window_start,
                end: window_end,
                slices: std::mem::take(&mut current),
            });
            window_start = slice.start;
            window_end = slice.end;
            current.push(slice);
        }
    }
    if !current.is_empty() {
        windows.push(PlannedWindow {
            start: window_start,
            end: window_end,
            slices: current,
        });
    }
    windows
}

pub(super) fn fetch_windows(
    windows: Vec<FetchWindow>,
) -> impl Stream<Item = Result<FetchedWindow>> + Send {
    stream::iter(windows)
        .map(fetch_window)
        .buffered(INGEST_FETCH_PARALLELISM)
}

pub(super) fn slice_coalesced_bytes(
    bytes: &Bytes,
    span_start: u64,
    slice: &ExternalSlice,
) -> Result<Bytes> {
    let relative = slice.start.checked_sub(span_start).ok_or_else(|| {
        Error::internal(format!(
            "External blob slice start {} is before coalesced read start {span_start}",
            slice.start
        ))
    })?;
    let offset = usize::try_from(relative).map_err(|_| {
        Error::invalid_input(format!(
            "External blob offset {relative} does not fit into usize"
        ))
    })?;
    let length = slice.end.checked_sub(slice.start).ok_or_else(|| {
        Error::invalid_input(format!(
            "External blob slice end {} is before start {}",
            slice.end, slice.start
        ))
    })?;
    let len = usize::try_from(length).map_err(|_| {
        Error::invalid_input(format!(
            "External blob length {length} does not fit into usize"
        ))
    })?;
    let Some(end) = offset.checked_add(len) else {
        return Err(Error::invalid_input(format!(
            "External blob slice offset {offset} + length {len} overflows usize"
        )));
    };
    if end > bytes.len() {
        return Err(Error::io(format!(
            "External blob slice {offset}..{end} exceeds coalesced read of {} bytes",
            bytes.len()
        )));
    }
    Ok(bytes.slice(offset..end))
}

async fn fetch_window(window: FetchWindow) -> Result<FetchedWindow> {
    let reader = window.open_reader.await?;
    let bytes = read_external_span(reader.as_ref(), window.start, window.end).await?;
    drop(reader);
    Ok(FetchedWindow {
        span_start: window.start,
        bytes,
        slices: window.slices,
    })
}

async fn read_external_span(reader: &dyn Reader, start: u64, end: u64) -> Result<Bytes> {
    let start_usize = usize::try_from(start).map_err(|_| {
        Error::invalid_input(format!(
            "External blob position {start} does not fit into usize"
        ))
    })?;
    let end_usize = usize::try_from(end).map_err(|_| {
        Error::invalid_input(format!(
            "External blob range end {end} does not fit into usize"
        ))
    })?;
    if end_usize < start_usize {
        return Err(Error::invalid_input(format!(
            "External blob range end {end} is before start {start}"
        )));
    }
    let bytes = reader
        .get_range(start_usize..end_usize)
        .await
        .map_err(Error::from)?;
    let expected = end_usize - start_usize;
    if bytes.len() != expected {
        return Err(Error::io(format!(
            "Short read for external blob range {start}..{end}: expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

pub(super) fn ingest_coalesce_hole(is_local: bool, block_size: u64) -> u64 {
    if is_local {
        block_size
    } else {
        object_store::OBJECT_STORE_COALESCE_DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use bytes::Bytes;

    use super::*;

    #[test]
    fn test_plan_coalesced_windows_merges_contiguous_slices_under_budget() {
        let slices = (0..1000)
            .map(|row| ExternalSlice {
                row,
                start: row as u64 * 8192,
                end: (row as u64 + 1) * 8192,
            })
            .collect();
        let windows = plan_coalesced_windows(slices, 64 * 1024, INGEST_COALESCE_BUDGET);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].slices.len(), 1000);
        assert_eq!(windows[0].start, 0);
        assert_eq!(windows[0].end, 1000 * 8192);
    }

    #[test]
    fn test_plan_coalesced_windows_keeps_wide_gaps_separate() {
        let gap = 1024 * 1024;
        let slices = (0..1000)
            .map(|row| {
                let start = row as u64 * (8192 + gap);
                ExternalSlice {
                    row,
                    start,
                    end: start + 8192,
                }
            })
            .collect();
        let windows = plan_coalesced_windows(slices, 64 * 1024, INGEST_COALESCE_BUDGET);
        assert_eq!(windows.len(), 1000);
    }

    #[test]
    fn test_plan_coalesced_windows_splits_when_span_exceeds_budget() {
        let slice_len = 3 * 1024 * 1024;
        let slices = (0..3)
            .map(|row| ExternalSlice {
                row,
                start: row as u64 * slice_len,
                end: (row as u64 + 1) * slice_len,
            })
            .collect();
        let windows = plan_coalesced_windows(
            slices,
            object_store::OBJECT_STORE_COALESCE_DEFAULT,
            INGEST_COALESCE_BUDGET,
        );
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].slices.len(), 2);
        assert_eq!(windows[1].slices.len(), 1);
        assert_eq!(windows[0].end, slice_len * 2);
        assert_eq!(windows[1].start, slice_len * 2);
    }

    #[test]
    fn test_ingest_coalesce_hole_is_block_size_on_local_and_one_mib_otherwise() {
        assert_eq!(ingest_coalesce_hole(true, 4 * 1024), 4 * 1024);
        assert_eq!(
            ingest_coalesce_hole(false, 64 * 1024),
            object_store::OBJECT_STORE_COALESCE_DEFAULT
        );
    }

    #[test]
    fn test_plan_coalesced_windows_merges_a_gap_equal_to_the_cloud_hole() {
        let hole = object_store::OBJECT_STORE_COALESCE_DEFAULT;
        let start = 100 + hole;
        let slices = vec![
            ExternalSlice {
                row: 0,
                start: 0,
                end: 100,
            },
            ExternalSlice {
                row: 1,
                start,
                end: start + 50,
            },
        ];
        let windows = plan_coalesced_windows(slices, hole, INGEST_COALESCE_BUDGET);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].slices.len(), 2);
        assert_eq!(windows[0].start, 0);
        assert_eq!(windows[0].end, start + 50);
    }

    #[test]
    fn test_plan_coalesced_windows_splits_a_gap_one_byte_past_the_cloud_hole() {
        let hole = object_store::OBJECT_STORE_COALESCE_DEFAULT;
        let start = 100 + hole + 1;
        let slices = vec![
            ExternalSlice {
                row: 0,
                start: 0,
                end: 100,
            },
            ExternalSlice {
                row: 1,
                start,
                end: start + 50,
            },
        ];
        let windows = plan_coalesced_windows(slices, hole, INGEST_COALESCE_BUDGET);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].end, 100);
        assert_eq!(windows[1].start, start);
    }

    #[test]
    fn test_plan_coalesced_windows_merges_overlapping_and_duplicate_slices() {
        let slices = vec![
            ExternalSlice {
                row: 2,
                start: 0,
                end: 20,
            },
            ExternalSlice {
                row: 1,
                start: 10,
                end: 30,
            },
            ExternalSlice {
                row: 0,
                start: 0,
                end: 20,
            },
        ];
        let windows = plan_coalesced_windows(slices, 0, INGEST_COALESCE_BUDGET);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start, 0);
        assert_eq!(windows[0].end, 30);
        assert_eq!(
            windows[0]
                .slices
                .iter()
                .map(|slice| slice.row)
                .collect::<Vec<_>>(),
            vec![0, 2, 1]
        );
    }

    #[test]
    fn test_plan_coalesced_windows_matches_explicit_spans() {
        type SpanCase<'a> = (&'a [(u64, u64)], u64, &'a [Range<u64>]);
        let cases: &[SpanCase] = &[
            (&[], 0, &[]),
            (&[(0, 3)], 0, &[0..3]),
            (&[(0, 2), (3, 5)], 0, &[0..2, 3..5]),
            (&[(0, 1), (1, 2)], 0, &[0..2]),
            (&[(0, 1), (2, 72)], 1, &[0..72]),
            (&[(0, 1), (56, 72), (73, 75)], 1, &[0..1, 56..75]),
            (&[(0, 1), (5, 6), (7, 9), (2, 3), (4, 6)], 1, &[0..9]),
            (
                &[(0, 1), (6, 7), (8, 9), (10, 14), (9, 10)],
                4,
                &[0..1, 6..14],
            ),
        ];
        for (ranges, hole, expected) in cases {
            let windows = plan(ranges, *hole, u64::MAX);
            assert_eq!(
                spans(&windows),
                expected.to_vec(),
                "ranges {ranges:?} hole {hole}"
            );
            assert_windows_are_split(&windows, *hole, u64::MAX);
            assert_payloads(ranges, *hole, u64::MAX);
        }
    }

    #[test]
    fn test_plan_coalesced_windows_fuzz_respects_hole_and_budget() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };
        for _ in 0..200 {
            let object_len = (next() % 240) + 10;
            let range_count = next() % 10;
            let hole = (next() % 5) + 1;
            let mut ranges = Vec::new();
            for _ in 0..range_count {
                let start = next() % object_len;
                let max_len = 20.min(object_len - start);
                let len = (next() % max_len) + 1;
                ranges.push((start, start + len));
            }
            let max_slice = ranges
                .iter()
                .map(|(start, end)| end - start)
                .max()
                .unwrap_or(1);
            let byte_budget = max_slice + (next() % 48);
            let windows = plan(&ranges, hole, byte_budget);
            let mut rows: Vec<_> = windows
                .iter()
                .flat_map(|window| window.slices.iter().map(|slice| slice.row))
                .collect();
            rows.sort_unstable();
            assert_eq!(rows, (0..ranges.len()).collect::<Vec<_>>());
            assert_windows_are_split(&windows, hole, byte_budget);
            assert_payloads(&ranges, hole, byte_budget);
        }
    }

    #[test]
    fn test_plan_coalesced_windows_merges_gap_within_block() {
        let slices = vec![
            ExternalSlice {
                row: 1,
                start: 10 + 32 * 1024,
                end: 10 + 32 * 1024 + 10,
            },
            ExternalSlice {
                row: 0,
                start: 0,
                end: 10,
            },
        ];
        let windows = plan_coalesced_windows(slices, 64 * 1024, INGEST_COALESCE_BUDGET);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].slices[0].row, 0);
        assert_eq!(windows[0].slices[1].row, 1);
        assert_eq!(windows[0].start, 0);
        assert_eq!(windows[0].end, 10 + 32 * 1024 + 10);
    }

    fn spans(windows: &[PlannedWindow]) -> Vec<Range<u64>> {
        windows
            .iter()
            .map(|window| window.start..window.end)
            .collect()
    }

    fn plan(ranges: &[(u64, u64)], hole: u64, byte_budget: u64) -> Vec<PlannedWindow> {
        let slices = ranges
            .iter()
            .enumerate()
            .map(|(row, (start, end))| ExternalSlice {
                row,
                start: *start,
                end: *end,
            })
            .collect();
        plan_coalesced_windows(slices, hole, byte_budget)
    }

    fn assert_payloads(ranges: &[(u64, u64)], hole: u64, byte_budget: u64) {
        let max_end = ranges.iter().map(|(_, end)| *end).max().unwrap_or(0) as usize;
        let file: Vec<u8> = (0..max_end).map(|index| index as u8).collect();
        for window in plan(ranges, hole, byte_budget) {
            let bytes = Bytes::from(file[window.start as usize..window.end as usize].to_vec());
            for slice in &window.slices {
                let payload = slice_coalesced_bytes(&bytes, window.start, slice).unwrap();
                assert_eq!(
                    payload.as_ref(),
                    &file[slice.start as usize..slice.end as usize]
                );
            }
        }
    }

    fn assert_windows_are_split(windows: &[PlannedWindow], hole: u64, byte_budget: u64) {
        for pair in windows.windows(2) {
            let current = &pair[0];
            let next = &pair[1].slices[0];
            let merged_end = current.end.max(next.end);
            let close = next.start <= current.end.saturating_add(hole);
            let within_budget = merged_end.saturating_sub(current.start) <= byte_budget;
            assert!(
                !(close && within_budget),
                "windows {:?} and {:?} could merge",
                current.start..current.end,
                pair[1].start..pair[1].end
            );
        }
        for window in windows {
            assert!(window.end.saturating_sub(window.start) <= byte_budget);
            assert_eq!(window.start, window.slices[0].start);
            assert_eq!(
                window.end,
                window.slices.iter().map(|slice| slice.end).max().unwrap()
            );
        }
    }
}
