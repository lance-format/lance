// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Validation inputs and storage-action semantics.
//!
//! Native transactions are validated against current fragment state before
//! lowering. The tree only replays the resulting state changes.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::format::pb::{self, fragment_action::Action};
use crate::format::{DataFile, Fragment};
use crate::fragment_metadata::action;
use lance_core::{Error, Result};
use object_store::path::Path;

/// Fragment state resolved from a snapshot, including known absent IDs.
///
/// `ids` records every id that was resolved, so an id that is absent from
/// `fragments` is known to be missing rather than unresolved. A commit may
/// only mutate ids in this set, or append ids the tree has never assigned.
#[derive(Debug, Clone, Default)]
pub struct TouchedFragments {
    pub ids: BTreeSet<u64>,
    pub fragments: BTreeMap<u64, Fragment>,
}

impl TouchedFragments {
    pub fn get(&self, fragment_id: u64) -> Option<&Fragment> {
        self.fragments.get(&fragment_id)
    }

    pub fn resolved(&self, fragment_id: u64) -> bool {
        self.ids.contains(&fragment_id)
    }
}

/// Storage actions from a validated native transaction. Tree preparation checks
/// them against [`TouchedFragments`] before storing them.
#[derive(Debug, Clone)]
pub struct ValidatedCommit {
    /// Advance the ID allocator, including reservations; never decrease it.
    pub next_fragment_id: Option<u64>,
    pub fragment_actions: Vec<pb::FragmentAction>,
}

impl ValidatedCommit {
    pub fn fragment_actions(fragment_actions: Vec<pb::FragmentAction>) -> Self {
        Self {
            next_fragment_id: None,
            fragment_actions,
        }
    }

    /// Advance the allocation frontier past upserts and explicit reservations.
    /// Aggregate validation has already checked the upserted fragment IDs.
    pub(super) fn allocation_frontier(&self, current: u64) -> Result<u64> {
        let mut frontier = current;
        for action in &self.fragment_actions {
            if let Some(Action::UpsertFragment(fragment)) = &action.action {
                let next = fragment.id.checked_add(1).ok_or_else(|| {
                    Error::invalid_input(format!("Fragment ID {} leaves no next ID", fragment.id))
                })?;
                frontier = frontier.max(next);
            }
        }
        let next = self.next_fragment_id.unwrap_or(frontier);
        if next < frontier || next > u64::from(u32::MAX) + 1 {
            return Err(Error::invalid_input(format!(
                "Fragment ID frontier {next} must be between {frontier} and 2^32"
            )));
        }
        Ok(next)
    }
}

/// Encode a data replacement with matching fields and file version, or a file
/// addition with disjoint fields, as file actions. Missing fragments and
/// unchanged replacements return an error, and so does a replacement file
/// actions cannot express. That is a partial field overlap, a fragment that
/// repeats a file path, a rename onto a path another matching file holds, or
/// an overlay on a replaced field, which only the transaction's read version
/// can resolve. Callers handling those must derive the final fragment and use
/// an upsert.
pub fn data_replacement(
    current: Option<&Fragment>,
    fragment_id: u64,
    replacement: &DataFile,
) -> Result<Vec<pb::FragmentAction>> {
    let fragment = current.ok_or_else(|| {
        Error::invalid_input(format!(
            "DataReplacement targets fragment {fragment_id} which does not exist"
        ))
    })?;
    let inexpressible = |reason: &str| {
        Error::invalid_input(format!(
            "DataReplacement for fragment {fragment_id} cannot be encoded as file actions: \
             {reason}, replacement {}",
            replacement.path
        ))
    };
    let mut paths = BTreeSet::new();
    if fragment.files.iter().any(|file| !paths.insert(&file.path)) {
        return Err(inexpressible("the fragment repeats a data file path"));
    }
    // An overlay committed after the transaction read its fragment outlives
    // the replacement, and one committed before does not.
    if fragment.overlays.iter().any(|overlay| {
        overlay
            .data_file
            .fields
            .iter()
            .any(|field| *field >= 0 && replacement.fields.contains(field))
    }) {
        return Err(inexpressible("an overlay covers a replaced field"));
    }
    let matching: Vec<&DataFile> = fragment
        .files
        .iter()
        .filter(|file| {
            file.fields == replacement.fields
                && file.file_major_version == replacement.file_major_version
                && file.file_minor_version == replacement.file_minor_version
        })
        .collect();
    if matching.is_empty() {
        let covered = fragment
            .files
            .iter()
            .flat_map(|file| file.fields.iter())
            .any(|field_id| replacement.fields.contains(field_id));
        if covered {
            return Err(Error::invalid_input(format!(
                "DataReplacement for fragment {fragment_id} partially overlaps existing fields: \
                 replacement fields={:?}",
                replacement.fields
            )));
        }
        return Ok(vec![action::add_data_file(fragment_id, replacement)]);
    }
    let unchanged = matching.iter().all(|file| {
        file.path == replacement.path
            && file.file_size_bytes == replacement.file_size_bytes
            && file.base_id == replacement.base_id
    });
    if unchanged {
        return Err(Error::invalid_input(format!(
            "DataReplacement for fragment {fragment_id} made no changes: the replacement \
             matches the existing data file {} exactly",
            replacement.path
        )));
    }
    if matching.len() > 1 && matching.iter().any(|file| file.path == replacement.path) {
        return Err(inexpressible(
            "another matching file already holds the replacement path",
        ));
    }
    Ok(matching
        .iter()
        .map(|file| action::replace_data_file(fragment_id, &file.path, replacement))
        .collect())
}

/// Validate action targets and derive their fragment and row-count deltas.
/// Unresolved IDs must be introduced by an upsert at or above `next_fragment_id`.
pub(crate) fn aggregate_deltas(
    actions: &[pb::FragmentAction],
    touched: &TouchedFragments,
    next_fragment_id: u64,
    base: &Path,
) -> Result<Vec<ActionDeltas>> {
    // Most actions only read their prior record, so copy the resolved
    // records only once an action has to be applied to them.
    let mut snapshot = Cow::Borrowed(&touched.fragments);
    let mut created = BTreeSet::new();
    let mut deltas = Vec::with_capacity(actions.len());
    let mut remaining = HashMap::<u64, usize>::with_capacity(actions.len());
    for fragment_action in actions {
        let fragment_id = action::target_frag_id(fragment_action).ok_or_else(|| {
            Error::invalid_input("fragment metadata tree commit contains an empty fragment action")
        })?;
        *remaining.entry(fragment_id).or_default() += 1;
    }
    for (offset, fragment_action) in actions.iter().enumerate() {
        let fragment_id = action::target_frag_id(fragment_action).ok_or_else(|| {
            Error::invalid_input("fragment metadata tree commit contains an empty fragment action")
        })?;
        let fresh_append = matches!(
            fragment_action.action,
            Some(Action::UpsertFragment(_)) if fragment_id >= next_fragment_id
        );
        if !fresh_append && !touched.resolved(fragment_id) && !created.contains(&fragment_id) {
            return Err(Error::invalid_input(format!(
                "fragment metadata tree commit mutates fragment {fragment_id} without resolving it first"
            )));
        }
        let before = snapshot.get(&fragment_id);
        let before = (
            i64::from(before.is_some()),
            physical_rows(before)?,
            visible_rows(before)?,
        );
        let last_for_fragment = remaining.get_mut(&fragment_id).is_some_and(|count| {
            *count -= 1;
            *count == 0
        });
        // A file appended by the last action on its fragment leaves every
        // count alone, so the record is not copied to apply it. The file is
        // still checked here, because once a drain writes it into a leaf,
        // every reader of that leaf would reject it.
        if let Some(Action::AddDataFile(add)) = &fragment_action.action
            && last_for_fragment
        {
            super::validation::action_target(fragment_action)?;
            if before.0 == 0 {
                return Err(Error::invalid_input(format!(
                    "AddDataFile action targets missing frag_id={fragment_id}"
                )));
            }
            if let Some(file) = &add.file {
                file.validate(base)?;
            }
            deltas.push(ActionDeltas {
                fragment_count: 0,
                physical_rows: 0,
                visible_rows: 0,
            });
            continue;
        }
        // A record no later action edits replaces its fragment whole, so it
        // is decoded on its own instead of into a copy of the resolved records.
        let mut replacement = BTreeMap::new();
        let state = match &fragment_action.action {
            Some(Action::UpsertFragment(_)) if last_for_fragment => &mut replacement,
            _ => snapshot.to_mut(),
        };
        crate::fragment_metadata::content::apply_actions(
            state,
            vec![pb::FragmentTreeMutation {
                action_sequence: offset as u64,
                action: Some(fragment_action.clone()),
                fragment_count_delta: 0,
                total_rows_delta: 0,
                visible_rows_delta: 0,
            }],
        )?;
        if fresh_append {
            created.insert(fragment_id);
        }
        let after = state.get(&fragment_id);
        if let Some(after) = after {
            require_storable(after, base)?;
        }
        deltas.push(ActionDeltas {
            fragment_count: i64::from(after.is_some()) - before.0,
            physical_rows: physical_rows(after)? - before.1,
            visible_rows: visible_rows(after)? - before.2,
        });
    }
    Ok(deltas)
}

/// Validate an initial inventory and return its allocation frontier, physical
/// row total, and visible row total. No tree objects are written on failure.
pub(super) fn validate_bootstrap(fragments: &[Fragment], base: &Path) -> Result<(u64, u64, u64)> {
    let mut total_rows = 0u64;
    let mut visible_rows = 0u64;
    let mut next_fragment_id = 0u64;
    for f in fragments {
        next_fragment_id = next_fragment_id.max(f.id.checked_add(1).ok_or_else(|| {
            Error::invalid_input(format!("Fragment ID {} leaves no next ID", f.id))
        })?);
        require_storable(f, base)?;
        visible_rows = visible_rows
            .checked_add(f.num_rows().unwrap_or(0) as u64)
            .ok_or_else(|| {
                Error::invalid_input(format!("Visible row count overflow at fragment {}", f.id))
            })?;
        total_rows = total_rows
            .checked_add(f.physical_rows.unwrap_or(0) as u64)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "fragment metadata tree total_rows overflow while bootstrapping fragment id={}",
                    f.id
                ))
            })?;
    }
    Ok((next_fragment_id, total_rows, visible_rows))
}

/// Require known physical and deletion counts for exact row aggregates, and
/// validate data files with the same checks used by the leaf decoder.
pub(crate) fn require_storable(fragment: &Fragment, base: &Path) -> Result<()> {
    if fragment.id > u64::from(u32::MAX) {
        return Err(Error::invalid_input(format!(
            "Fragment ID {} exceeds u32",
            fragment.id
        )));
    }
    action::require_known_counts(fragment)?;
    if let (Some(rows), Some(deleted)) = (
        fragment.physical_rows,
        fragment
            .deletion_file
            .as_ref()
            .and_then(|file| file.num_deleted_rows),
    ) && deleted > rows
    {
        return Err(Error::invalid_input(format!(
            "fragment {} has {deleted} deleted rows but only {rows} physical rows",
            fragment.id
        )));
    }
    for file in fragment.referenced_lance_files() {
        file.validate(base)?;
    }
    Ok(())
}

/// One action's contribution to the tree aggregates, derived by applying it
/// to the touched snapshot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ActionDeltas {
    pub fragment_count: i64,
    pub physical_rows: i64,
    /// Physical minus deleted rows, exact by [`require_storable`].
    pub visible_rows: i64,
}

/// A fragment's visible rows. An absent fragment contributes zero; counts
/// are known by [`require_storable`], enforced before this is read.
fn visible_rows(fragment: Option<&Fragment>) -> Result<i64> {
    let Some(fragment) = fragment else {
        return Ok(0);
    };
    let rows = fragment.num_rows().ok_or_else(|| {
        Error::internal(format!(
            "fragment {} reached visible-row accounting with unknown counts; \
             require_storable must run first",
            fragment.id
        ))
    })?;
    to_i64(rows)
}

fn to_i64(rows: usize) -> Result<i64> {
    i64::try_from(rows).map_err(|_| {
        Error::invalid_input(format!(
            "row count does not fit aggregate delta: rows={rows}"
        ))
    })
}

fn physical_rows(fragment: Option<&Fragment>) -> Result<i64> {
    let rows = fragment
        .and_then(|fragment| fragment.physical_rows)
        .unwrap_or(0);
    i64::try_from(rows).map_err(|_| {
        Error::invalid_input(format!(
            "fragment physical_rows does not fit aggregate delta: physical_rows={rows}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment_metadata::support::{
        make_backfill_data_file, make_fragment, make_replacement_data_file,
    };
    use rstest::rstest;

    #[rstest]
    #[case::past_address_space(u64::from(u32::MAX) + 2)]
    #[case::maximum_u64(u64::MAX)]
    fn allocation_frontier_rejects_invalid_current(#[case] current: u64) {
        let commit = ValidatedCommit::fragment_actions(Vec::new());
        let error = commit.allocation_frontier(current).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains(&current.to_string()), "{message}");
        assert!(message.contains("2^32"), "{message}");
    }

    #[test]
    fn allocation_frontier_preserves_exhausted_address_space() {
        let current = u64::from(u32::MAX) + 1;
        let commit = ValidatedCommit::fragment_actions(Vec::new());

        assert_eq!(commit.allocation_frontier(current).unwrap(), current);
    }

    #[test]
    fn data_replacement_encodes_matching_or_disjoint_fields() {
        let fragment = make_fragment(7);

        // Exact fields and file version: swap the named file in place.
        let matching = make_replacement_data_file(7, 0);
        let actions = data_replacement(Some(&fragment), 7, &matching).unwrap();
        assert_eq!(actions.len(), 1);
        match actions[0].action.as_ref().unwrap() {
            Action::ReplaceDataFile(replace) => {
                assert_eq!(replace.expected_path, fragment.files[0].path);
                assert_eq!(replace.path, matching.path);
            }
            other => panic!("expected ReplaceDataFile, got {other:?}"),
        }

        // File actions match by first path, so duplicate paths need a whole
        // record from the caller.
        let mut aliased = fragment.clone();
        aliased.files.push(aliased.files[0].clone());
        let error = data_replacement(Some(&aliased), 7, &matching).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("repeats a data file path"),
            "{error}"
        );

        // Disjoint fields: the all-NULL add-column case appends verbatim.
        let disjoint = make_backfill_data_file(7, 0);
        let actions = data_replacement(Some(&fragment), 7, &disjoint).unwrap();
        assert!(matches!(
            actions[0].action.as_ref().unwrap(),
            Action::AddDataFile(add) if add.frag_id == 7
        ));

        // Identical file: rejected as a no-op.
        let error = data_replacement(Some(&fragment), 7, &fragment.files[0]).unwrap_err();
        assert!(error.to_string().contains("no changes"), "{error}");

        // Partial field overlap: rejected.
        let mut overlapping = make_replacement_data_file(7, 1);
        overlapping.fields = vec![1, 99].into();
        let error = data_replacement(Some(&fragment), 7, &overlapping).unwrap_err();
        assert!(error.to_string().contains("partially overlaps"), "{error}");

        // Missing fragment: rejected.
        let error = data_replacement(None, 99, &matching).unwrap_err();
        assert!(error.to_string().contains("fragment 99"), "{error}");
    }

    #[test]
    fn require_storable_rejects_data_files_the_leaf_decoder_rejects() {
        let mut fragment = make_fragment(4);
        fragment.files[0].column_indices = Vec::new().into();

        let error = require_storable(&fragment, &Path::from("table")).unwrap_err();

        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("contained fewer column_indices than fields"),
            "{error}"
        );
    }

    #[test]
    fn aggregate_deltas_keep_counts_for_a_last_add_data_file() {
        let touched = TouchedFragments {
            ids: BTreeSet::from([3, 9]),
            fragments: BTreeMap::from([(3, make_fragment(3))]),
        };
        let file = make_backfill_data_file(3, 0);
        let deltas = aggregate_deltas(
            &[action::add_data_file(3, &file)],
            &touched,
            10,
            &Path::default(),
        )
        .unwrap();
        assert_eq!(deltas.len(), 1);
        assert_eq!(
            (
                deltas[0].fragment_count,
                deltas[0].physical_rows,
                deltas[0].visible_rows
            ),
            (0, 0, 0)
        );
        // A later action on the same fragment still sees the appended file.
        let deltas = aggregate_deltas(
            &[
                action::add_data_file(3, &file),
                action::remove_data_file(3, file.path.clone()),
                action::remove_fragment(3),
            ],
            &touched,
            10,
            &Path::default(),
        )
        .unwrap();
        assert_eq!(
            deltas
                .iter()
                .map(|delta| delta.fragment_count)
                .collect::<Vec<_>>(),
            [0, 0, -1]
        );
        // Resolved as absent: the file has no record to attach to.
        let error = aggregate_deltas(
            &[action::add_data_file(9, &file)],
            &touched,
            10,
            &Path::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("missing frag_id=9"), "{error}");
        let empty = pb::FragmentAction {
            action: Some(Action::AddDataFile(pb::AddDataFile {
                frag_id: 3,
                file: None,
            })),
        };
        let error = aggregate_deltas(&[empty], &touched, 10, &Path::default()).unwrap_err();
        assert!(error.to_string().contains("has no file"), "{error}");
    }

    #[test]
    fn aggregate_deltas_reject_unresolved_mutations_and_stale_appends() {
        let touched = TouchedFragments::default();
        let error = aggregate_deltas(
            &[action::remove_fragment(3)],
            &touched,
            10,
            &Path::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("without resolving"), "{error}");

        let error = aggregate_deltas(
            &[action::upsert_fragment(&make_fragment(3)).unwrap()],
            &touched,
            10,
            &Path::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("without resolving"), "{error}");

        let deltas = aggregate_deltas(
            &[action::upsert_fragment(&make_fragment(10)).unwrap()],
            &touched,
            10,
            &Path::default(),
        )
        .unwrap();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].fragment_count, 1);
        assert_eq!(deltas[0].physical_rows, 1);
        assert_eq!(deltas[0].visible_rows, 1);

        let empty = make_fragment(10).with_physical_rows(0);
        let deltas = aggregate_deltas(
            &[action::upsert_fragment(&empty).unwrap()],
            &touched,
            10,
            &Path::default(),
        )
        .unwrap();
        assert_eq!(deltas[0].physical_rows, 0);
    }

    #[test]
    fn aggregate_deltas_rejects_malformed_files_a_last_action_carries() {
        let touched = TouchedFragments {
            ids: BTreeSet::from([3]),
            fragments: BTreeMap::from([(3, make_fragment(3))]),
        };
        let mut appended = make_backfill_data_file(3, 0);
        appended.column_indices = Vec::new().into();
        let mut upserted = make_fragment(10);
        upserted.files[0].column_indices = Vec::new().into();
        let mut overlaid = make_fragment(11);
        overlaid
            .overlays
            .push(crate::format::overlay::DataOverlayFile {
                data_file: appended.clone(),
                coverage: crate::format::overlay::OverlayCoverage::dense(
                    roaring::RoaringBitmap::from_iter([0]),
                ),
                committed_version: 1,
            });
        let cases = [
            (action::add_data_file(3, &appended), appended.path.clone()),
            (
                action::upsert_fragment(&upserted).unwrap(),
                upserted.files[0].path.clone(),
            ),
            (
                action::upsert_fragment(&overlaid).unwrap(),
                appended.path.clone(),
            ),
        ];

        for (fragment_action, path) in cases {
            let error = aggregate_deltas(&[fragment_action], &touched, 10, &Path::from("table"))
                .unwrap_err();

            assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
            // Error paths percent-encode the separator, so match the file name.
            let file_name = path.rsplit('/').next().unwrap();
            let message = error.to_string();
            assert!(message.contains(file_name), "{message}");
            assert!(
                message.contains("contained fewer column_indices than fields"),
                "{message}"
            );
        }
    }

    /// The resolved record is intact, so input it cannot take is the caller's
    /// error, not corrupt metadata.
    #[rstest]
    #[case::overcounted_deletion(
        pb::FragmentAction {
            action: Some(Action::AddDeletionFile(pb::AddDeletionFile {
                frag_id: 7,
                deletion_file: Some(pb::DeletionFile {
                    read_version: 3,
                    id: 11,
                    file_type: pb::deletion_file::DeletionFileType::Bitmap.into(),
                    num_deleted_rows: 3,
                    base_id: None,
                }),
            })),
        },
        "fragment 7 has 3 deleted rows but only 2 physical rows"
    )]
    #[case::replacement_of_an_absent_file(
        action::replace_data_file(7, "data/absent.lance", &make_replacement_data_file(7, 0)),
        "data/absent.lance"
    )]
    fn aggregate_deltas_rejects_input_the_resolved_record_cannot_take(
        #[case] fragment_action: pb::FragmentAction,
        #[case] expected: &str,
    ) {
        let touched = TouchedFragments {
            ids: BTreeSet::from([7]),
            fragments: BTreeMap::from([(7, make_fragment(7).with_physical_rows(2))]),
        };

        let error =
            aggregate_deltas(&[fragment_action], &touched, 10, &Path::default()).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(expected), "{error}");
    }
}
