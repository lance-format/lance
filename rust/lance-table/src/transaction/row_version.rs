// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Helpers for resolving and encoding per-row dataset-version metadata used by
//! update-style transactions.

use std::collections::{HashMap, HashSet};

use crate::format::{
    Fragment, RowDatasetVersionMeta, RowDatasetVersionRun, RowDatasetVersionSequence, RowIdMeta,
};
use crate::rowids::read_row_ids;
use crate::rowids::segment::U64Segment;
use crate::rowids::version::build_version_meta;
use lance_core::{Error, Result};

/// Fallback version for rows whose original creation version cannot be determined.
/// Version 1 is the initial dataset version in the Lance format.
pub const UNKNOWN_CREATED_AT_VERSION: u64 = 1;

/// Look up the `created_at` version for a single UPDATE-branch row ID.
///
/// Callers must only call this for row IDs that are confirmed to be present in
/// `row_id_to_source` (i.e. UPDATE branch rows whose source exists in an existing
/// fragment).  INSERT branch rows (no source) must use `new_version` directly and
/// must not call this function.
///
/// Uses `row_id_to_source` to find the originating fragment and row offset, then
/// performs a O(K) random-access lookup via [`RowDatasetVersionSequence::version_at`]
/// on the pre-decoded sequence in `version_cache` (keyed by fragment ID).
///
/// Returns [`UNKNOWN_CREATED_AT_VERSION`] if the source fragment has no
/// `created_at_version_meta` (missing or failed to decode) or the offset is
/// out of range.
fn resolve_created_at_version(
    row_id: u64,
    row_id_to_source: &HashMap<u64, (&Fragment, usize)>,
    version_cache: &HashMap<u64, RowDatasetVersionSequence>,
) -> u64 {
    let Some((orig_frag, row_offset)) = row_id_to_source.get(&row_id) else {
        return UNKNOWN_CREATED_AT_VERSION;
    };
    let Some(seq) = version_cache.get(&orig_frag.id) else {
        return UNKNOWN_CREATED_AT_VERSION;
    };
    seq.version_at(*row_offset)
        .unwrap_or(UNKNOWN_CREATED_AT_VERSION)
}

/// For each new fragment produced by an update, set `created_at_version_meta`
/// (preserved from the original rows) and `last_updated_at_version_meta`.
pub fn resolve_update_version_metadata(
    existing_fragments: &[Fragment],
    new_fragments: &mut [Fragment],
    new_version: u64,
) -> Result<()> {
    // Collect only the row IDs we actually need to resolve, those appearing in new_fragments
    // with inline metadata. This bounds the lookup map to O(updated rows) instead of O(all dataset rows)
    let needed_row_ids: HashSet<u64> = new_fragments
        .iter()
        .filter_map(|f| match &f.row_id_meta {
            Some(RowIdMeta::Inline(data)) => read_row_ids(data).ok(),
            _ => None,
        })
        .flat_map(|seq| seq.iter().collect::<Vec<_>>())
        .collect();

    let mut row_id_to_source: HashMap<u64, (&Fragment, usize)> = HashMap::new();

    if !needed_row_ids.is_empty() {
        // Compute the bounding range of the needed set once.  Any fragment whose
        // entire row-id range lies outside [needed_min, needed_max] cannot contain
        // any needed ID and can be skipped before the inner per-row loop.
        let needed_min = *needed_row_ids.iter().min().unwrap();
        let needed_max = *needed_row_ids.iter().max().unwrap();

        // Stable row IDs must be globally unique among *live* rows, but after a rewrite-style
        // update the same stable ID can appear twice in `existing_fragments`: once in an older
        // fragment's inline `row_id_meta` at the original row offset (rows may be soft-deleted
        // via a deletion vector) and again in a newer fragment holding rewritten data. For
        // `created_at` we need the mapping from the original fragment/offset; that is always the
        // first occurrence when fragments are processed in ascending `id` order.
        let mut sorted_frags: Vec<&Fragment> = existing_fragments.iter().collect();
        sorted_frags.sort_by_key(|f| f.id);
        for frag in sorted_frags {
            if let Some(RowIdMeta::Inline(data)) = &frag.row_id_meta
                && let Ok(seq) = read_row_ids(data)
            {
                // Range pre-filter: skip the per-row inner loop when the fragment's
                // bounding row-id range has no overlap with [needed_min, needed_max].
                // row_id_range() returns None for empty sequences, which are also skipped.
                // This is a conservative check (may produce false positives for sparse
                // segments) but never skips a fragment that actually contains a needed ID.
                if seq
                    .row_id_range()
                    .is_none_or(|r| *r.end() < needed_min || *r.start() > needed_max)
                {
                    continue;
                }

                for (offset, rid) in seq.iter().enumerate() {
                    if needed_row_ids.contains(&rid) {
                        row_id_to_source.entry(rid).or_insert((frag, offset));
                    }
                }
            }
        }
    }

    // Pre-decode the `created_at` version sequence for each source fragment exactly
    // once.  Without this cache, resolve_created_at_version would call load_sequence()
    // (a protobuf decode) for every single updated row, even when many rows originate
    // from the same fragment.
    let source_frag_ids: HashSet<u64> = row_id_to_source.values().map(|(f, _)| f.id).collect();
    let version_cache: HashMap<u64, RowDatasetVersionSequence> = existing_fragments
        .iter()
        .filter(|f| source_frag_ids.contains(&f.id))
        .filter_map(|frag| {
            let seq = frag
                .created_at_version_meta
                .as_ref()?
                .load_sequence()
                .ok()?;
            Some((frag.id, seq))
        })
        .collect();

    for fragment in new_fragments.iter_mut() {
        let row_ids = match &fragment.row_id_meta {
            Some(RowIdMeta::Inline(data)) => read_row_ids(data).ok(),
            Some(RowIdMeta::External(_)) => {
                log::warn!(
                    "Fragment {} has external row ID metadata; \
                     version tracking will use defaults",
                    fragment.id,
                );
                None
            }
            None => None,
        };

        if let Some(row_ids) = row_ids {
            let physical_rows = fragment.physical_rows.unwrap_or(0);
            let created_at_versions: Vec<u64> = row_ids
                .iter()
                .map(|rid| {
                    if row_id_to_source.contains_key(&rid) {
                        // UPDATE branch: stable row ID resolves to a source row in an
                        // existing fragment.  Copy created_at from the original row so
                        // the row's first-appearance version is preserved across rewrites.
                        resolve_created_at_version(rid, &row_id_to_source, &version_cache)
                    } else {
                        // INSERT branch: stable row ID has no source in existing fragments
                        // (e.g. NOT MATCHED arm of MERGE INTO).  The row first appears in
                        // this commit, so created_at equals the new commit version.
                        new_version
                    }
                })
                .collect();
            debug_assert_eq!(created_at_versions.len(), physical_rows);

            let runs = encode_version_runs(&created_at_versions);
            let created_at_seq = RowDatasetVersionSequence { runs };
            fragment.created_at_version_meta = Some(
                RowDatasetVersionMeta::from_sequence(&created_at_seq).map_err(|e| {
                    Error::internal(format!(
                        "Failed to create created_at version metadata: {}",
                        e
                    ))
                })?,
            );

            fragment.last_updated_at_version_meta = build_version_meta(fragment, new_version);
        } else {
            let version_meta = build_version_meta(fragment, new_version);
            fragment.last_updated_at_version_meta = version_meta.clone();
            fragment.created_at_version_meta = version_meta;
        }
    }
    Ok(())
}

/// Whether `fragment` is a "pure rewrite" fragment produced by a `RewriteRows`
/// update — every row carries a stable row id (its inline `row_id_meta` count
/// equals `physical_rows`), so the fragment holds no freshly inserted rows.
///
/// This is the per-fragment test `Transaction::collect_pure_rewrite_row_update_frags_ids`
/// runs; it is also called from `update_to_actions` to classify a
/// pre-id-assignment `Operation::Update` `new_fragments` entry.
pub fn is_pure_rewrite_fragment(fragment: &Fragment) -> Result<bool> {
    let physical_rows = fragment
        .physical_rows
        .ok_or_else(|| Error::internal("Fragment does not have physical rows"))?
        as u64;
    let existing_row_count = match &fragment.row_id_meta {
        Some(RowIdMeta::Inline(data)) => read_row_ids(data)?.len() as u64,
        Some(_) => 0,
        None => return Ok(false),
    };
    Ok(existing_row_count == physical_rows)
}

/// Run-length encode a sequence of per-row versions into [`RowDatasetVersionRun`]s.
pub fn encode_version_runs(versions: &[u64]) -> Vec<RowDatasetVersionRun> {
    if versions.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut current_version = versions[0];
    let mut run_start = 0u64;
    for (i, &version) in versions.iter().enumerate().skip(1) {
        if version != current_version {
            runs.push(RowDatasetVersionRun {
                span: U64Segment::Range(run_start..i as u64),
                version: current_version,
            });
            current_version = version;
            run_start = i as u64;
        }
    }
    runs.push(RowDatasetVersionRun {
        span: U64Segment::Range(run_start..versions.len() as u64),
        version: current_version,
    });
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_version_runs_empty() {
        let runs = encode_version_runs(&[]);
        assert!(runs.is_empty());
    }

    #[test]
    fn test_encode_version_runs_single_run() {
        let runs = encode_version_runs(&[3, 3, 3]);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].version, 3);
    }

    #[test]
    fn test_encode_version_runs_alternating() {
        let runs = encode_version_runs(&[1, 2, 1, 2]);
        assert_eq!(runs.len(), 4);
        assert_eq!(runs[0].version, 1);
        assert_eq!(runs[1].version, 2);
        assert_eq!(runs[2].version, 1);
        assert_eq!(runs[3].version, 2);
    }
}
