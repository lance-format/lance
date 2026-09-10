// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row ids and the per-row version metadata that travels with them.
//!
//! Under stable row ids each fragment carries two run-length encoded sequences:
//! `created_at_version`, stamped once when a row first appears, and
//! `last_updated_at_version`, refreshed whenever a row's values change. Keeping
//! `created_at` correct across an update means tracing each new row back to the
//! fragment and offset it came from, which is what most of this module does.

use crate::format::{
    Fragment, RowDatasetVersionMeta, RowDatasetVersionRun, RowDatasetVersionSequence, RowIdMeta,
};
use crate::rowids::segment::U64Segment;
use crate::rowids::version::build_version_meta;
use crate::rowids::{RowIdSequence, read_row_ids, write_row_ids};
use crate::transaction::Transaction;
use lance_core::{Error, Result};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// Fallback version for rows whose original creation version cannot be determined.
/// Version 1 is the initial dataset version in the Lance format.
const UNKNOWN_CREATED_AT_VERSION: u64 = 1;

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
pub(super) fn resolve_update_version_metadata(
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

/// Run-length encode a sequence of per-row versions into [`RowDatasetVersionRun`]s.
fn encode_version_runs(versions: &[u64]) -> Vec<RowDatasetVersionRun> {
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

impl Transaction {
    /// collect the pure(the num of row IDs are equal to the physical rows) "rewrite rows" updated fragment ids
    pub(super) fn collect_pure_rewrite_row_update_frags_ids(
        fragments: &[Fragment],
    ) -> Result<Vec<u64>> {
        let mut pure_update_frag_ids = Vec::new();

        for fragment in fragments {
            let physical_rows = fragment
                .physical_rows
                .ok_or_else(|| Error::internal("Fragment does not have physical rows"))?
                as u64;

            if let Some(row_id_meta) = &fragment.row_id_meta {
                let existing_row_count = match row_id_meta {
                    RowIdMeta::Inline(data) => {
                        let sequence = read_row_ids(data)?;
                        sequence.len() as u64
                    }
                    _ => 0,
                };

                // only filter the fragments that match: all the rows have row id,
                // which means it does not contain inserted rows in this fragment
                if existing_row_count == physical_rows {
                    pure_update_frag_ids.push(fragment.id);
                }
            }
        }

        Ok(pure_update_frag_ids)
    }

    pub(super) fn assign_row_ids(next_row_id: &mut u64, fragments: &mut [Fragment]) -> Result<()> {
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
                        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized.into()));
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
                fragment.row_id_meta = Some(RowIdMeta::Inline(serialized.into()));
                *next_row_id += physical_rows;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::test_support::{
        created_at_versions, default_build_config, last_updated_at_versions,
        make_stable_row_id_manifest, update_txn,
    };
    use std::sync::Arc;

    #[test]
    fn test_assign_row_ids_new_fragment() {
        // Test assigning row IDs to a fragment without existing row IDs
        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: Some(100),
            row_id_meta: None,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 0;

        Transaction::assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        assert_eq!(next_row_id, 100);
        assert!(fragments[0].row_id_meta.is_some());

        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 100);
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, (0..100).collect::<Vec<u64>>());
        } else {
            panic!("Expected inline row ID metadata");
        }
    }

    #[test]
    fn test_assign_row_ids_existing_complete() {
        // Test with fragment that already has complete row IDs
        let existing_sequence = RowIdSequence::from(0..50);
        let serialized = write_row_ids(&existing_sequence);

        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: Some(50),
            row_id_meta: Some(RowIdMeta::Inline(serialized.into())),
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 100;

        Transaction::assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        // next_row_id should not change
        assert_eq!(next_row_id, 100);

        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 50);
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, (0..50).collect::<Vec<u64>>());
        } else {
            panic!("Expected inline row ID metadata");
        }
    }

    #[test]
    fn test_assign_row_ids_partial_existing() {
        // Test with fragment that has partial row IDs (merge insert case)
        let existing_sequence = RowIdSequence::from(0..30);
        let serialized = write_row_ids(&existing_sequence);

        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: Some(50), // More physical rows than existing row IDs
            row_id_meta: Some(RowIdMeta::Inline(serialized.into())),
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 100;

        Transaction::assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        // next_row_id should advance by 20 (50 - 30)
        assert_eq!(next_row_id, 120);

        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 50);
            let row_ids: Vec<u64> = sequence.iter().collect();
            // Should contain original 0-29 plus new 100-119
            let mut expected = (0..30).collect::<Vec<u64>>();
            expected.extend(100..120);
            assert_eq!(row_ids, expected);
        } else {
            panic!("Expected inline row ID metadata");
        }
    }

    #[test]
    fn test_assign_row_ids_excess_row_ids() {
        // Test error case where fragment has more row IDs than physical rows
        let existing_sequence = RowIdSequence::from(0..60);
        let serialized = write_row_ids(&existing_sequence);

        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: Some(50), // Less physical rows than existing row IDs
            row_id_meta: Some(RowIdMeta::Inline(serialized.into())),
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 100;

        let result = Transaction::assign_row_ids(&mut next_row_id, &mut fragments);

        assert!(result.is_err());
        if let Err(Error::Internal { message, .. }) = result {
            assert!(message.contains("more row IDs (60) than physical rows (50)"));
        } else {
            panic!("Expected Internal error about excess row IDs");
        }
    }

    #[test]
    fn test_assign_row_ids_multiple_fragments() {
        // Test with multiple fragments, some with existing row IDs, some without
        let existing_sequence = RowIdSequence::from(500..520);
        let serialized = write_row_ids(&existing_sequence);

        let mut fragments = vec![
            Fragment {
                id: 1,
                physical_rows: Some(30), // No existing row IDs
                row_id_meta: None,
                files: vec![],
                overlays: vec![],
                deletion_file: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
            },
            Fragment {
                id: 2,
                physical_rows: Some(25), // Partial existing row IDs
                row_id_meta: Some(RowIdMeta::Inline(serialized.into())),
                files: vec![],
                overlays: vec![],
                deletion_file: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
            },
        ];
        let mut next_row_id = 1000;

        Transaction::assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        // Should advance by 30 (first fragment) + 5 (second fragment partial)
        assert_eq!(next_row_id, 1035);

        // Check first fragment
        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 30);
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, (1000..1030).collect::<Vec<u64>>());
        } else {
            panic!("Expected inline row ID metadata for first fragment");
        }

        // Check second fragment
        if let Some(RowIdMeta::Inline(data)) = &fragments[1].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 25);
            let row_ids: Vec<u64> = sequence.iter().collect();
            // Should contain original 500-519 plus new 1030-1034
            let mut expected = (500..520).collect::<Vec<u64>>();
            expected.extend(1030..1035);
            assert_eq!(row_ids, expected);
        } else {
            panic!("Expected inline row ID metadata for second fragment");
        }
    }

    #[test]
    fn test_assign_row_ids_missing_physical_rows() {
        // Test error case where fragment doesn't have physical_rows set
        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: None,
            row_id_meta: None,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 0;

        let result = Transaction::assign_row_ids(&mut next_row_id, &mut fragments);

        assert!(result.is_err());
        if let Err(Error::Internal { message, .. }) = result {
            assert!(message.contains("Fragment does not have physical rows"));
        } else {
            panic!("Expected Internal error about missing physical rows");
        }
    }

    #[test]
    fn test_update_version_tracking_preserves_created_at() {
        let existing_seq = RowIdSequence::from([100u64, 101, 102].as_slice());
        let created_at_seq = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..3),
                version: 5,
            }],
        };
        let existing_fragment = Fragment {
            id: 1,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&existing_seq).into())),
            physical_rows: Some(3),
            created_at_version_meta: Some(
                RowDatasetVersionMeta::from_sequence(&created_at_seq).unwrap(),
            ),
            last_updated_at_version_meta: None,
        };

        let new_seq = RowIdSequence::from([100u64, 102].as_slice());
        let new_fragment = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let manifest = make_stable_row_id_manifest(vec![existing_fragment]);
        let (result, _) = update_txn(vec![new_fragment])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        assert_eq!(created_at_versions(&result, 10), vec![5, 5]);
        assert_eq!(last_updated_at_versions(&result, 10), vec![5, 5]);
    }

    #[test]
    fn test_update_version_tracking_mixed_origins() {
        let frag_a_seq = RowIdSequence::from([10u64, 11].as_slice());
        let frag_a_created = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..2),
                version: 2,
            }],
        };
        let frag_b_seq = RowIdSequence::from([20u64, 21, 22].as_slice());
        let frag_b_created = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..3),
                version: 3,
            }],
        };

        let manifest = make_stable_row_id_manifest(vec![
            Fragment {
                id: 1,
                files: vec![],
                overlays: vec![],
                deletion_file: None,
                row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&frag_a_seq).into())),
                physical_rows: Some(2),
                created_at_version_meta: Some(
                    RowDatasetVersionMeta::from_sequence(&frag_a_created).unwrap(),
                ),
                last_updated_at_version_meta: None,
            },
            Fragment {
                id: 2,
                files: vec![],
                overlays: vec![],
                deletion_file: None,
                row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&frag_b_seq).into())),
                physical_rows: Some(3),
                created_at_version_meta: Some(
                    RowDatasetVersionMeta::from_sequence(&frag_b_created).unwrap(),
                ),
                last_updated_at_version_meta: None,
            },
        ]);

        // New fragment has rows from both original fragments: row 11 from frag_a, row 20 from frag_b
        let new_seq = RowIdSequence::from([11u64, 20].as_slice());
        let new_fragment = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let (result, _) = update_txn(vec![new_fragment])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        // Row 11 came from frag_a (offset 1, version 2), row 20 came from frag_b (offset 0, version 3)
        assert_eq!(created_at_versions(&result, 10), vec![2, 3]);
        assert_eq!(last_updated_at_versions(&result, 10), vec![5, 5]);
    }

    #[test]
    fn test_update_version_tracking_insert_branch_gets_new_version() {
        // Simulates the INSERT branch (NOT MATCHED) of a MERGE INTO commit:
        // the new fragment contains a mix of rewritten rows (UPDATE branch, row ID
        // present in existing fragments) and freshly inserted rows (INSERT branch,
        // row ID not present in any existing fragment).
        //
        // UPDATE branch row (10): created_at must be copied from the source fragment.
        // INSERT branch row (999): created_at must equal new_version (the merge commit
        // version), because the row first appeared in this commit.
        let existing_seq = RowIdSequence::from([10u64, 11].as_slice());
        let existing_created = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..2),
                version: 5,
            }],
        };
        let existing_fragment = Fragment {
            id: 1,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&existing_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: Some(
                RowDatasetVersionMeta::from_sequence(&existing_created).unwrap(),
            ),
            last_updated_at_version_meta: None,
        };

        // New fragment has row 10 (UPDATE branch) and row 999 (INSERT branch)
        let new_seq = RowIdSequence::from([10u64, 999].as_slice());
        let new_fragment = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        // update_txn uses read_version 4 → new_version is 5
        let manifest = make_stable_row_id_manifest(vec![existing_fragment]);
        let (result, _) = update_txn(vec![new_fragment])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        // Row 10 (UPDATE branch): created_at copied from source (version 5).
        // Row 999 (INSERT branch): created_at == new_version (5).
        assert_eq!(created_at_versions(&result, 10), vec![5, 5]);
        assert_eq!(last_updated_at_versions(&result, 10), vec![5, 5]);
    }

    #[test]
    fn test_update_version_tracking_merge_into_distinguishes_insert_and_update_branch() {
        // Verifies the MERGE INTO correctness contract when UPDATE branch rows and INSERT
        // branch rows have *different* source created_at values, so we can distinguish
        // which row got which value.
        //
        // Existing fragment (id=1): row IDs [10, 11], created_at = version 3.
        // New fragment (id=20): row IDs [10, 500, 11, 501].
        //   - Rows 10 and 11: UPDATE branch (present in existing fragment) → created_at = 3.
        //   - Rows 500 and 501: INSERT branch (no source) → created_at = new_version = 5.
        let existing_seq = RowIdSequence::from([10u64, 11].as_slice());
        let existing_created = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..2),
                version: 3,
            }],
        };
        let existing_fragment = Fragment {
            id: 1,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&existing_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: Some(
                RowDatasetVersionMeta::from_sequence(&existing_created).unwrap(),
            ),
            last_updated_at_version_meta: None,
        };

        let new_seq = RowIdSequence::from([10u64, 500, 11, 501].as_slice());
        let new_fragment = Fragment {
            id: 20,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(4),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        // update_txn uses read_version 4 → new_version is 5
        let manifest = make_stable_row_id_manifest(vec![existing_fragment]);
        let (result, _) = update_txn(vec![new_fragment])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        // UPDATE branch rows (10, 11): created_at preserved from source (version 3).
        // INSERT branch rows (500, 501): created_at == new_version (5).
        assert_eq!(created_at_versions(&result, 20), vec![3, 5, 3, 5]);
        // All rows in the new fragment get last_updated == new_version.
        assert_eq!(last_updated_at_versions(&result, 20), vec![5, 5, 5, 5]);
    }

    #[test]
    fn test_update_version_tracking_source_fragment_no_created_at_defaults_to_1() {
        // Source fragment has row_id_meta but no created_at_version_meta.
        // The row IS found in the lookup, but the version defaults to 1.
        let existing_seq = RowIdSequence::from([50u64, 51].as_slice());
        let existing_fragment = Fragment {
            id: 1,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&existing_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let new_seq = RowIdSequence::from([50u64].as_slice());
        let new_fragment = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(1),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let manifest = make_stable_row_id_manifest(vec![existing_fragment]);
        let (result, _) = update_txn(vec![new_fragment])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        // Row 50 is found in source but source has no created_at_version_meta → default 1
        assert_eq!(created_at_versions(&result, 10), vec![1]);
        assert_eq!(last_updated_at_versions(&result, 10), vec![5]);
    }

    #[test]
    fn test_update_version_tracking_no_row_id_meta_fallback() {
        let existing_seq = RowIdSequence::from([10u64, 11].as_slice());
        let existing_fragment = Fragment {
            id: 1,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&existing_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let new_fragment = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(3),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let manifest = make_stable_row_id_manifest(vec![existing_fragment]);
        let (result, _) = update_txn(vec![new_fragment])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        // Fragment starts with no row_id_meta → assign_row_ids gives it fresh IDs →
        // those IDs have no source in existing fragments (INSERT branch) →
        // created_at == new_version (5) for each row.
        assert_eq!(created_at_versions(&result, 10), vec![5, 5, 5]);
        assert_eq!(last_updated_at_versions(&result, 10), vec![5, 5, 5]);
    }

    #[test]
    fn test_update_version_tracking_corrupt_created_at_defaults_to_1() {
        let existing_seq = RowIdSequence::from([10u64, 11].as_slice());
        let existing_fragment = Fragment {
            id: 1,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&existing_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: Some(RowDatasetVersionMeta::Inline(Arc::from(
                vec![0xFFu8; 8].as_slice(),
            ))),
            last_updated_at_version_meta: None,
        };

        let new_seq = RowIdSequence::from([10u64].as_slice());
        let new_fragment = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(1),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let manifest = make_stable_row_id_manifest(vec![existing_fragment]);
        let (result, _) = update_txn(vec![new_fragment])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        // Corrupt metadata causes decode to fail → falls back to UNKNOWN_CREATED_AT_VERSION (1)
        assert_eq!(created_at_versions(&result, 10), vec![1]);
        assert_eq!(last_updated_at_versions(&result, 10), vec![5]);
    }

    /// Fragments whose row-ID range lies entirely outside the needed set must not
    /// affect the result.  Here fragment 1 has IDs [1000, 1001] which are far above
    /// the needed range [10, 11]; it is skipped by the range pre-filter and its
    /// created_at version (version 99) must never appear in the output.
    #[test]
    fn test_update_version_tracking_range_filter_skips_non_overlapping_fragment() {
        // Fragment in range – IDs [10, 11], created_at = 5
        let in_range_seq = RowIdSequence::from([10u64, 11].as_slice());
        let in_range_created = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..2),
                version: 5,
            }],
        };
        let in_range_frag = Fragment {
            id: 1,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&in_range_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: Some(
                RowDatasetVersionMeta::from_sequence(&in_range_created).unwrap(),
            ),
            last_updated_at_version_meta: None,
        };

        // Fragment outside range – IDs [1000, 1001], created_at = 99 (must never appear)
        let out_of_range_seq = RowIdSequence::from([1000u64, 1001].as_slice());
        let out_of_range_created = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..2),
                version: 99,
            }],
        };
        let out_of_range_frag = Fragment {
            id: 2,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&out_of_range_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: Some(
                RowDatasetVersionMeta::from_sequence(&out_of_range_created).unwrap(),
            ),
            last_updated_at_version_meta: None,
        };

        // New fragment rewrites both rows from the in-range fragment
        let new_seq = RowIdSequence::from([10u64, 11].as_slice());
        let new_frag = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let manifest = make_stable_row_id_manifest(vec![in_range_frag, out_of_range_frag]);
        let (result, _) = update_txn(vec![new_frag])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        // Both rows originate from the in-range fragment (version 5).
        // The out-of-range fragment's version 99 must not appear.
        assert_eq!(created_at_versions(&result, 10), vec![5, 5]);
        assert_eq!(last_updated_at_versions(&result, 10), vec![5, 5]);
    }

    /// When the needed row IDs fall exactly at the boundary of a fragment's range,
    /// the range pre-filter must NOT skip the fragment (boundary values are inclusive).
    #[test]
    fn test_update_version_tracking_range_filter_boundary_inclusive() {
        // Fragment IDs [10, 11, 12], created_at = 7
        let seq = RowIdSequence::from([10u64, 11, 12].as_slice());
        let created = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..3),
                version: 7,
            }],
        };
        let existing = Fragment {
            id: 1,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&seq).into())),
            physical_rows: Some(3),
            created_at_version_meta: Some(RowDatasetVersionMeta::from_sequence(&created).unwrap()),
            last_updated_at_version_meta: None,
        };

        // New fragment takes the boundary IDs: 10 (min) and 12 (max)
        let new_seq = RowIdSequence::from([10u64, 12].as_slice());
        let new_frag = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let manifest = make_stable_row_id_manifest(vec![existing]);
        let (result, _) = update_txn(vec![new_frag])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        // Boundary IDs must be found and resolved correctly
        assert_eq!(created_at_versions(&result, 10), vec![7, 7]);
    }

    /// When multiple updated rows all originate from the same source fragment,
    /// the created_at version sequence for that fragment must be decoded exactly
    /// once (not once per row).  The observable correctness requirement is that
    /// all rows get the right version regardless of how many there are.
    #[test]
    fn test_update_version_tracking_many_rows_same_source_fragment() {
        // Source fragment: 100 rows with IDs 0..100, mixed versions (2 runs).
        // First 50 rows at version 3, next 50 rows at version 4.
        let src_ids: Vec<u64> = (0u64..100).collect();
        let src_seq = RowIdSequence::from(src_ids.as_slice());
        let src_created = RowDatasetVersionSequence {
            runs: vec![
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..50),
                    version: 3,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..50),
                    version: 4,
                },
            ],
        };
        let src_frag = Fragment {
            id: 1,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&src_seq).into())),
            physical_rows: Some(100),
            created_at_version_meta: Some(
                RowDatasetVersionMeta::from_sequence(&src_created).unwrap(),
            ),
            last_updated_at_version_meta: None,
        };

        // New fragment rewrites all 100 rows preserving their stable IDs.
        let new_seq = RowIdSequence::from(src_ids.as_slice());
        let new_frag = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(100),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let manifest = make_stable_row_id_manifest(vec![src_frag]);
        let (result, _) = update_txn(vec![new_frag])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let versions = created_at_versions(&result, 10);
        assert_eq!(versions.len(), 100);
        // First 50 rows came from version 3, next 50 from version 4
        assert!(versions[..50].iter().all(|&v| v == 3));
        assert!(versions[50..].iter().all(|&v| v == 4));
    }

    /// Rows originating from multiple distinct source fragments must each get
    /// the version from their own source, even when all cached together.
    #[test]
    fn test_update_version_tracking_cache_multiple_source_fragments() {
        let seq_a = RowIdSequence::from([10u64, 11, 12].as_slice());
        let created_a = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..3),
                version: 2,
            }],
        };
        let seq_b = RowIdSequence::from([20u64, 21, 22].as_slice());
        let created_b = RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..3),
                version: 8,
            }],
        };

        let manifest = make_stable_row_id_manifest(vec![
            Fragment {
                id: 1,
                files: vec![],
                overlays: vec![],
                deletion_file: None,
                row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&seq_a).into())),
                physical_rows: Some(3),
                created_at_version_meta: Some(
                    RowDatasetVersionMeta::from_sequence(&created_a).unwrap(),
                ),
                last_updated_at_version_meta: None,
            },
            Fragment {
                id: 2,
                files: vec![],
                overlays: vec![],
                deletion_file: None,
                row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&seq_b).into())),
                physical_rows: Some(3),
                created_at_version_meta: Some(
                    RowDatasetVersionMeta::from_sequence(&created_b).unwrap(),
                ),
                last_updated_at_version_meta: None,
            },
        ]);

        // New fragment takes rows from both sources: 12 (frag A, offset 2) and 20 (frag B, offset 0)
        let new_seq = RowIdSequence::from([12u64, 20].as_slice());
        let new_frag = Fragment {
            id: 10,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&new_seq).into())),
            physical_rows: Some(2),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
        };

        let (result, _) = update_txn(vec![new_frag])
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        // Row 12 → frag A offset 2 → version 2; row 20 → frag B offset 0 → version 8
        assert_eq!(created_at_versions(&result, 10), vec![2, 8]);
    }

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
