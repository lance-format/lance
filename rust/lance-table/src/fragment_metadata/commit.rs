// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Validation inputs and storage-action semantics.
//!
//! A [`Transaction`](crate::transaction::Transaction) is validated against
//! current fragment state before it becomes storage actions. The tree only
//! replays those actions.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::format::pb::{self, fragment_action::Action};
use crate::format::{DataFile, Fragment};
use crate::fragment_metadata::{action, node};
use lance_core::{Error, Result};
use object_store::path::Path;

/// Fragment state resolved from a snapshot, including known absent IDs.
///
/// `ids` records every id that was resolved, so an id that is absent from
/// `fragments` is known to be missing rather than unresolved.
#[derive(Debug, Clone, Default)]
pub struct TouchedFragments {
    pub ids: BTreeSet<u64>,
    pub fragments: BTreeMap<u64, Fragment>,
}

impl TouchedFragments {
    /// The fragment, or `None` when it is absent or not yet resolved.
    /// [`Self::ids`] tells those apart.
    pub fn get(&self, fragment_id: u64) -> Option<&Fragment> {
        self.fragments.get(&fragment_id)
    }
}

/// Storage actions from a [`Transaction`](crate::transaction::Transaction),
/// which owns operation validation. Tree preparation checks them against
/// [`TouchedFragments`] before storing them.
#[derive(Debug, Clone)]
pub struct FragmentChanges {
    /// The `next_fragment_id` these changes publish. `None` derives it from the
    /// tree's current value and the upserted IDs. `Some` reserves IDs up to the
    /// given value, and preparation rejects a value below the derived one or
    /// above 2^32.
    pub(super) next_fragment_id: Option<u64>,
    pub(super) fragment_actions: Vec<pb::FragmentAction>,
}

impl FragmentChanges {
    /// Changes that reserve no fragment IDs beyond the ones they upsert.
    pub fn new(fragment_actions: Vec<pb::FragmentAction>) -> Self {
        Self {
            next_fragment_id: None,
            fragment_actions,
        }
    }

    /// Publish this `next_fragment_id`, reserving IDs past the upserts.
    /// Preparation refuses a value below the one the upserts need or above
    /// 2^32.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentChanges;
    /// let reservation = FragmentChanges::new(Vec::new()).with_next_fragment_id(8);
    /// # let _ = reservation;
    /// ```
    pub fn with_next_fragment_id(mut self, next_fragment_id: u64) -> Self {
        self.next_fragment_id = Some(next_fragment_id);
        self
    }

    /// Return the `next_fragment_id` this commit publishes, raised past its
    /// upserts and any reservation. Refuse an action targeting that value or
    /// above, because readers reject it. Aggregate validation refuses every
    /// other edit of an absent record, so only a removal reaches this check,
    /// since it replays as a no-op on an absent record.
    pub(super) fn checked_next_fragment_id(&self, current: u64) -> Result<u64> {
        let mut minimum_next = current;
        for action in &self.fragment_actions {
            if let Some(Action::UpsertFragment(fragment)) = &action.action {
                let next = fragment.id.checked_add(1).ok_or_else(|| {
                    Error::invalid_input(format!("Fragment ID {} leaves no next ID", fragment.id))
                })?;
                minimum_next = minimum_next.max(next);
            }
        }
        let next = self.next_fragment_id.unwrap_or(minimum_next);
        if next < minimum_next || next > node::ROOT_EXCLUSIVE_END {
            return Err(Error::invalid_input(format!(
                "next_fragment_id {next} must be at least {minimum_next} and at most 2^32"
            )));
        }
        let unallocated = self
            .fragment_actions
            .iter()
            .filter_map(action::target_frag_id)
            .find(|target| *target >= next);
        if let Some(target) = unallocated {
            return Err(Error::invalid_input(format!(
                "fragment metadata tree commit mutates unallocated fragment {target}, \
                 at or above next_fragment_id {next}"
            )));
        }
        Ok(next)
    }
}

/// Encode a data replacement as file actions. The replacement either matches a
/// file's fields and version, or adds a file with disjoint fields.
///
/// Returns an error when the replacement changes nothing, or when file actions
/// can't express it. That happens for a partial field overlap, a fragment that
/// repeats a file path, a rename onto a path another matching file holds, or a
/// replaced field that has an overlay, which only the transaction's read
/// version can resolve. For those, callers derive the final fragment and use an
/// upsert.
///
/// ```
/// # use lance_table::format::DataFile;
/// # use lance_table::fragment_metadata::{TouchedFragments, data_replacement_actions};
/// # fn example(touched: &TouchedFragments, file: &DataFile) -> lance_core::Result<()> {
/// if let Some(fragment) = touched.get(3) {
///     let actions = data_replacement_actions(fragment, file)?;
/// }
/// # Ok(())
/// # }
/// ```
pub fn data_replacement_actions(
    fragment: &Fragment,
    replacement: &DataFile,
) -> Result<Vec<pb::FragmentAction>> {
    let fragment_id = fragment.id;
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

/// The checks shared with the read path report a bad action or file as corrupt
/// metadata. In a commit the bad input came from the caller, not from storage,
/// so turn it into invalid input and keep the file path in the message.
fn as_commit_input(error: Error) -> Error {
    match error {
        Error::CorruptFile { path, source, .. } => {
            Error::invalid_input(format!("{path}: {source}"))
        }
        error => error,
    }
}

/// Validate action targets and derive their fragment and row-count deltas.
pub(super) fn aggregate_deltas(
    actions: &[pb::FragmentAction],
    fragments: &BTreeMap<u64, Fragment>,
    base: &Path,
) -> Result<Vec<ActionDeltas>> {
    // Most actions only read their prior record, so copy the resolved
    // records only once an action has to be applied to them.
    let mut snapshot = Cow::Borrowed(fragments);
    let mut deltas = Vec::with_capacity(actions.len());
    let mut remaining = HashMap::<u64, usize>::with_capacity(actions.len());
    for fragment_action in actions {
        let fragment_id = action::target_frag_id(fragment_action).ok_or_else(|| {
            Error::invalid_input("fragment metadata tree commit contains an empty fragment action")
        })?;
        *remaining.entry(fragment_id).or_default() += 1;
    }
    for fragment_action in actions {
        let fragment_id = action::target_frag_id(fragment_action).ok_or_else(|| {
            Error::invalid_input("fragment metadata tree commit contains an empty fragment action")
        })?;
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
        // A file is checked before either path applies it, because once a
        // drain writes it into a leaf, every reader of that leaf would reject it.
        if let Some(Action::AddDataFile(add)) = &fragment_action.action
            && let Some(file) = &add.file
        {
            file.validate(base).map_err(as_commit_input)?;
        }
        // A file appended by the last action on its fragment leaves every
        // count alone, so the record is not copied to apply it.
        if matches!(fragment_action.action, Some(Action::AddDataFile(_))) && last_for_fragment {
            super::validation::action_target(fragment_action).map_err(as_commit_input)?;
            if before.0 == 0 {
                return Err(Error::invalid_input(format!(
                    "AddDataFile action targets missing frag_id={fragment_id}"
                )));
            }
            deltas.push(ActionDeltas {
                fragment_count: 0,
                total_rows: 0,
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
            std::slice::from_ref(fragment_action),
        )?;
        let after = state.get(&fragment_id);
        if let Some(after) = after {
            // Only an upsert brings a whole record. Any other action leaves
            // resolved files in place, and replay does not validate files from
            // a stored buffer, so one that fails here is corrupt metadata.
            let storable = require_storable(after, base);
            match fragment_action.action {
                Some(Action::UpsertFragment(_)) => storable.map_err(as_commit_input)?,
                _ => storable?,
            }
        }
        deltas.push(ActionDeltas {
            fragment_count: i64::from(after.is_some()) - before.0,
            total_rows: physical_rows(after)? - before.1,
            visible_rows: visible_rows(after)? - before.2,
        });
    }
    Ok(deltas)
}

/// Validate an initial inventory and return its next_fragment_id. No tree
/// objects are written on failure.
pub(super) fn validate_initial_fragments(fragments: &[Fragment], base: &Path) -> Result<u64> {
    let mut next_fragment_id = 0u64;
    for f in fragments {
        next_fragment_id = next_fragment_id.max(f.id.checked_add(1).ok_or_else(|| {
            Error::invalid_input(format!("Fragment ID {} leaves no next ID", f.id))
        })?);
        require_storable(f, base)?;
    }
    Ok(next_fragment_id)
}

/// Require known physical and deletion counts for exact row aggregates, and
/// validate data files with the same checks used by the leaf decoder.
fn require_storable(fragment: &Fragment, base: &Path) -> Result<()> {
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
pub(super) struct ActionDeltas {
    pub fragment_count: i64,
    pub total_rows: i64,
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
    fn checked_next_fragment_id_rejects_invalid_current(#[case] current: u64) {
        let changes = FragmentChanges::new(Vec::new());
        let error = changes.checked_next_fragment_id(current).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains(&current.to_string()), "{message}");
        assert!(message.contains("2^32"), "{message}");
    }

    #[test]
    fn checked_next_fragment_id_preserves_exhausted_address_space() {
        let current = u64::from(u32::MAX) + 1;
        let changes = FragmentChanges::new(Vec::new());

        assert_eq!(changes.checked_next_fragment_id(current).unwrap(), current);
    }

    #[test]
    fn data_replacement_actions_swaps_the_file_with_matching_fields() {
        let fragment = make_fragment(7);
        let matching = make_replacement_data_file(7, 0);
        let actions = data_replacement_actions(&fragment, &matching).unwrap();
        assert_eq!(actions.len(), 1);
        match actions[0].action.as_ref().unwrap() {
            Action::ReplaceDataFile(replace) => {
                assert_eq!(replace.expected_path, fragment.files[0].path);
                assert_eq!(replace.path, matching.path);
            }
            other => panic!("expected ReplaceDataFile, got {other:?}"),
        }
    }

    #[test]
    fn data_replacement_actions_appends_a_file_with_disjoint_fields() {
        let fragment = make_fragment(7);
        // The all-NULL add-column case appends verbatim.
        let disjoint = make_backfill_data_file(7, 0);
        let actions = data_replacement_actions(&fragment, &disjoint).unwrap();
        assert!(matches!(
            actions[0].action.as_ref().unwrap(),
            Action::AddDataFile(add) if add.frag_id == 7
        ));
    }

    #[rstest]
    // File actions match by first path, so duplicate paths need a whole record
    // from the caller.
    #[case::repeated_path(
        Fragment {
            files: vec![make_fragment(7).files[0].clone(); 2],
            ..make_fragment(7)
        },
        make_replacement_data_file(7, 0),
        "repeats a data file path"
    )]
    #[case::identical_file(
        make_fragment(7),
        make_fragment(7).files[0].clone(),
        "no changes"
    )]
    #[case::partial_overlap(
        make_fragment(7),
        DataFile {
            fields: vec![1, 99].into(),
            ..make_replacement_data_file(7, 1)
        },
        "partially overlaps"
    )]
    #[case::rename_onto_a_held_path(
        Fragment {
            files: [make_fragment(7).files, vec![make_replacement_data_file(7, 0)]].concat(),
            ..make_fragment(7)
        },
        make_replacement_data_file(7, 0),
        "already holds the replacement path"
    )]
    fn data_replacement_actions_rejects_invalid_replacements(
        #[case] fragment: Fragment,
        #[case] replacement: DataFile,
        #[case] expected: &str,
    ) {
        let error = data_replacement_actions(&fragment, &replacement).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(expected), "{error}");
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
        let fragments = BTreeMap::from([(3, make_fragment(3))]);
        let file = make_backfill_data_file(3, 0);
        let deltas = aggregate_deltas(
            &[action::add_data_file(3, &file)],
            &fragments,
            &Path::default(),
        )
        .unwrap();
        assert_eq!(deltas.len(), 1);
        assert_eq!(
            (
                deltas[0].fragment_count,
                deltas[0].total_rows,
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
            &fragments,
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
    }

    #[test]
    fn aggregate_deltas_rejects_an_add_data_file_without_a_file() {
        let fragments = BTreeMap::from([(3, make_fragment(3))]);
        let empty = pb::FragmentAction {
            action: Some(Action::AddDataFile(pb::AddDataFile {
                frag_id: 3,
                file: None,
            })),
        };

        // The fast path takes a last action, and replay takes the same action
        // when a later one edits the fragment. Both report the same class.
        for actions in [vec![empty.clone()], vec![empty, action::remove_fragment(3)]] {
            let error = aggregate_deltas(&actions, &fragments, &Path::default()).unwrap_err();

            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(error.to_string().contains("has no file"), "{error}");
        }
    }

    /// The action names nothing wrong, so a file that fails on the resolved
    /// record came from storage.
    #[test]
    fn aggregate_deltas_reports_a_malformed_resolved_file_as_corrupt() {
        let mut resolved = make_fragment(3);
        resolved.files[0].column_indices = Vec::new().into();
        let fragments = BTreeMap::from([(3, resolved)]);

        let error = aggregate_deltas(
            &[action::clear_deletion_file(3)],
            &fragments,
            &Path::default(),
        )
        .unwrap_err();

        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("contained fewer column_indices than fields"),
            "{error}"
        );
    }

    #[test]
    fn aggregate_deltas_count_an_upsert_of_an_absent_fragment() {
        let fragments = BTreeMap::new();
        let deltas = aggregate_deltas(
            &[action::upsert_fragment(&make_fragment(10)).unwrap()],
            &fragments,
            &Path::default(),
        )
        .unwrap();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].fragment_count, 1);
        assert_eq!(deltas[0].total_rows, 1);
        assert_eq!(deltas[0].visible_rows, 1);

        let empty = make_fragment(10).with_physical_rows(0);
        let deltas = aggregate_deltas(
            &[action::upsert_fragment(&empty).unwrap()],
            &fragments,
            &Path::default(),
        )
        .unwrap();
        assert_eq!(deltas[0].total_rows, 0);
    }

    #[test]
    fn aggregate_deltas_rejects_malformed_files_a_commit_carries() {
        let fragments = BTreeMap::from([(3, make_fragment(3))]);
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
            (
                vec![action::add_data_file(3, &appended)],
                appended.path.clone(),
            ),
            (
                vec![
                    action::add_data_file(3, &appended),
                    action::remove_fragment(3),
                ],
                appended.path.clone(),
            ),
            (
                vec![action::upsert_fragment(&upserted).unwrap()],
                upserted.files[0].path.clone(),
            ),
            (
                vec![action::upsert_fragment(&overlaid).unwrap()],
                appended.path.clone(),
            ),
        ];

        for (actions, path) in cases {
            let error = aggregate_deltas(&actions, &fragments, &Path::from("table")).unwrap_err();

            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
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
                    inline: None,
                }),
            })),
        },
        "fragment 7 has 3 deleted rows but only 2 physical rows"
    )]
    #[case::replacement_of_an_absent_file(
        action::replace_data_file(7, "data/absent.lance", &make_replacement_data_file(7, 0)),
        "data/absent.lance"
    )]
    #[case::add_data_file_on_an_absent_fragment(
        action::add_data_file(9, &make_backfill_data_file(9, 0)),
        "missing frag_id=9"
    )]
    fn aggregate_deltas_rejects_input_the_resolved_record_cannot_take(
        #[case] fragment_action: pb::FragmentAction,
        #[case] expected: &str,
    ) {
        let fragments = BTreeMap::from([(7, make_fragment(7).with_physical_rows(2))]);

        let error = aggregate_deltas(&[fragment_action], &fragments, &Path::default()).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(expected), "{error}");
    }
}
