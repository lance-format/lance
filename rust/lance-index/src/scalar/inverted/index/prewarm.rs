// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// Target on-disk size of one posting-list read chunk. Keep this large enough
/// that cloud stores do not spend bulk reads on thousands of tiny ranges, but
/// still bounded so one large partition is not materialized all at once.
pub(super) const POSTING_READ_CHUNK_TARGET_BYTES: u64 = 128 << 20;

/// Cap on token rows per chunk, bounding the built `Vec` when posting lists are tiny.
pub(super) const POSTING_READ_MAX_CHUNK_TOKENS: usize = 256 * 1024;

/// Floor on token rows per chunk, so a partition always makes progress.
pub(super) const POSTING_READ_MIN_CHUNK_TOKENS: usize = 1;

/// Maximum number of child values materialized in any 32-bit-offset posting
/// column. The planner accounts for the widest selected column (impacts when
/// present), keeping Arrow's `List<i32>` offsets representable.
pub(super) const POSTING_READ_MAX_LIST_CHILDREN: u64 = i32::MAX as u64;

/// Token rows per chunk: byte target / average bytes-per-token, clamped to `[MIN, MAX]`.
pub(super) fn posting_read_chunk_tokens(token_count: usize, file_size_bytes: u64) -> usize {
    if token_count == 0 {
        return POSTING_READ_MIN_CHUNK_TOKENS;
    }
    let bytes_per_token = (file_size_bytes / token_count as u64).max(1); // >= 1: no div-by-zero
    let by_bytes = (POSTING_READ_CHUNK_TARGET_BYTES / bytes_per_token) as usize;
    by_bytes.clamp(POSTING_READ_MIN_CHUNK_TOKENS, POSTING_READ_MAX_CHUNK_TOKENS)
}

pub(super) fn synthetic_group_aligned_chunk_end(
    group_size: usize,
    token_count: usize,
    tok_start: usize,
    desired_end: usize,
) -> usize {
    if desired_end >= token_count {
        return token_count;
    }

    let boundary = desired_end - (desired_end % group_size);
    if boundary > tok_start {
        boundary
    } else {
        tok_start.saturating_add(group_size).min(token_count)
    }
}

pub(super) fn synthetic_group_ranges_for_chunk(
    group_size: usize,
    tok_start: usize,
    tok_end: usize,
    token_count: usize,
) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();
    let mut start = tok_start - (tok_start % group_size);
    if start < tok_start {
        start = start.saturating_add(group_size).min(token_count);
    }
    while start < tok_end {
        let end = start.saturating_add(group_size).min(token_count);
        ranges.push((
            u32::try_from(start).unwrap_or(u32::MAX),
            u32::try_from(end).unwrap_or(u32::MAX),
        ));
        start = end;
    }
    ranges
}

pub(super) fn prewarm_chunk_ranges(
    grouping: &PostingGrouping,
    token_count: usize,
    chunk_tokens: usize,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut tok_start = 0usize;
    while tok_start < token_count {
        let mut tok_end = (tok_start + chunk_tokens).min(token_count);
        // `tok_start` is always a group boundary; snap `tok_end` back to one too.
        if grouping.is_grouped() {
            tok_end = grouping.aligned_chunk_end(token_count, tok_start, tok_end);
        }
        ranges.push((tok_start, tok_end));
        tok_start = tok_end;
    }
    ranges
}
