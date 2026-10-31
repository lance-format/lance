// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fragment mutation replay, reduction, and row accounting.

use std::collections::{BTreeMap, BTreeSet};

use lance_core::{Error, Result};

use super::action;
use super::store::NodeStore;
use crate::format::Fragment;
use crate::format::pb::{self, fragment_action::Action};

/// Target fragment ID of an action validated before routing.
pub(super) fn action_key(mutation: &pb::FragmentTreeMutation) -> u64 {
    mutation
        .action
        .as_ref()
        .and_then(action::target_frag_id)
        .unwrap_or(0)
}

pub(super) fn apply_aggregate_delta(base: u64, delta: i64, name: &str) -> Result<u64> {
    if delta >= 0 {
        base.checked_add(delta as u64).ok_or_else(|| {
            Error::invalid_input(format!(
                "fragment metadata tree {name} aggregate overflow: base={base}, delta={delta}"
            ))
        })
    } else {
        base.checked_sub(delta.unsigned_abs()).ok_or_else(|| {
            Error::invalid_input(format!(
                "fragment metadata tree {name} aggregate underflow: base={base}, delta={delta}"
            ))
        })
    }
}

pub(super) fn sum_aggregate_deltas(
    deltas: impl IntoIterator<Item = i64>,
    name: &str,
) -> Result<i64> {
    deltas.into_iter().try_fold(0i64, |sum, delta| {
        sum.checked_add(delta).ok_or_else(|| {
            Error::invalid_input(format!(
                "fragment metadata tree {name} overflow while summing: sum={sum}, delta={delta}"
            ))
        })
    })
}

pub(super) fn sum_aggregate_values(
    values: impl IntoIterator<Item = u64>,
    name: &str,
) -> Result<u64> {
    values.into_iter().try_fold(0u64, |sum, value| {
        sum.checked_add(value).ok_or_else(|| {
            Error::invalid_input(format!(
                "fragment metadata tree {name} overflow while summing: sum={sum}, value={value}"
            ))
        })
    })
}

pub(super) fn fragment_row_counts(fragments: &[Fragment]) -> Result<(u64, u64)> {
    let total_rows = sum_aggregate_values(
        fragments
            .iter()
            .map(|fragment| fragment.physical_rows.unwrap_or(0) as u64),
        "total_rows",
    )?;
    // Bootstrap refuses unknown counts and decoding gives every other record
    // known ones, so the default is never taken.
    let visible_rows = sum_aggregate_values(
        fragments
            .iter()
            .map(|fragment| fragment.num_rows().unwrap_or(0) as u64),
        "visible_rows",
    )?;
    Ok((total_rows, visible_rows))
}

/// Whether a buffered action removes a fragment the subtree below still
/// holds. A squashed append-then-remove keeps a removal that retires nothing.
pub(super) fn retires_fragment(mutation: &pb::FragmentTreeMutation) -> bool {
    mutation.fragment_count_delta < 0
        && matches!(
            mutation
                .action
                .as_ref()
                .and_then(|action| action.action.as_ref()),
            Some(Action::RemoveFragment(_))
        )
}

/// Stored fragments that draining these actions removes.
pub(super) fn retired_keys(actions: &[pb::FragmentTreeMutation]) -> u64 {
    actions
        .iter()
        .filter(|action| retires_fragment(action))
        .count() as u64
}

/// Whether an action supplies a complete record, independently of whether it
/// adds a fragment or overwrites one already stored below.
fn upserts_fragment(mutation: &pb::FragmentTreeMutation) -> bool {
    matches!(
        mutation
            .action
            .as_ref()
            .and_then(|action| action.action.as_ref()),
        Some(Action::UpsertFragment(_))
    )
}

/// Whether a buffered action overwrites a record the subtree below holds.
/// After squashing, a whole-record upsert that does not change the fragment
/// count replaces a stored record.
fn replaces_fragment(mutation: &pb::FragmentTreeMutation) -> bool {
    mutation.fragment_count_delta == 0 && upserts_fragment(mutation)
}

/// Replay one leaf, reading only headers when every stored record is replaced or removed.
pub(super) async fn replay_leaf(
    store: &NodeStore,
    child: &pb::FragmentTreeChild,
    end: u64,
    incoming: Vec<pb::FragmentTreeMutation>,
) -> Result<Vec<Fragment>> {
    let retired = retired_keys(&incoming);
    let replaced = incoming
        .iter()
        .filter(|action| replaces_fragment(action))
        .count() as u64;
    let map: BTreeMap<u64, Fragment> = if retired + replaced == child.num_keys {
        // Headers verify the deltas and that every surviving record
        // is replaced in full.
        let overwritten: BTreeSet<u64> = incoming
            .iter()
            .filter(|action| upserts_fragment(action))
            .map(action_key)
            .collect();
        let mut map: BTreeMap<u64, Fragment> = store
            .read_leaf_headers(child, end)
            .await?
            .into_iter()
            .map(|header| (header.id, header))
            .collect();
        let stored: Vec<u64> = map.keys().copied().collect();
        store.apply_verified(&mut map, incoming)?;
        if let Some(id) = stored
            .iter()
            .find(|id| map.contains_key(id) && !overwritten.contains(id))
        {
            return Err(super::validation::corrupt(format!(
                "Buffered actions for leaf {} count every stored fragment as \
                     removed or replaced, but fragment {id} is neither",
                child.path
            )));
        }
        store.share_data_file_lists(map.values_mut());
        map
    } else {
        let fragments = store.read_leaf_in_range(child, end).await?;
        let mut map: BTreeMap<u64, Fragment> = fragments.into_iter().map(|f| (f.id, f)).collect();
        store.apply_verified(&mut map, incoming)?;
        map
    };
    Ok(map.into_values().collect())
}

/// Whether replay checks each mutation's count deltas against the record it
/// changes. Storage paths verify. The reducer oracle and lowering compare
/// records only, with deltas left unset.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DeltaCheck {
    Trust,
    Verify,
}

impl DeltaCheck {
    /// Verified actions were read from storage, so one that replay cannot
    /// apply is corrupt metadata. Trusted actions are commit input, so it is
    /// the caller's error.
    fn failure(self, error: Error) -> Error {
        let message = match error {
            Error::CorruptFile { source, .. }
            | Error::InvalidInput { source, .. }
            | Error::NotSupported { source, .. } => source.to_string(),
            error => return error,
        };
        match self {
            Self::Trust => Error::invalid_input(message),
            Self::Verify => super::validation::corrupt(message),
        }
    }
}

/// Apply buffered mutations to an id-keyed fragment map in `(key, action_sequence)` order.
/// `action_sequence` is assigned at commit time and only ever grows, so this
/// order is the commit order per fragment wherever actions meet.
pub fn apply_actions(
    frags: &mut BTreeMap<u64, Fragment>,
    actions: Vec<pb::FragmentTreeMutation>,
) -> Result<()> {
    apply_mutations(frags, actions, DeltaCheck::Trust)
}

/// Apply mutations and reject any whose stored deltas differ from the change
/// they make. The record before and after is in hand here, so this is where
/// the format's exact-delta rule is enforced.
pub fn apply_verified(
    frags: &mut BTreeMap<u64, Fragment>,
    actions: Vec<pb::FragmentTreeMutation>,
) -> Result<()> {
    apply_mutations(frags, actions, DeltaCheck::Verify)
}

fn record_counts(fragment: Option<&Fragment>) -> (i64, i64, i64) {
    fragment.map_or((0, 0, 0), |fragment| {
        (
            1,
            fragment.physical_rows.unwrap_or(0) as i64,
            fragment.num_rows().unwrap_or(0) as i64,
        )
    })
}

fn apply_mutations(
    frags: &mut BTreeMap<u64, Fragment>,
    actions: Vec<pb::FragmentTreeMutation>,
    check: DeltaCheck,
) -> Result<()> {
    replay(frags, actions, check).map_err(|error| check.failure(error))
}

fn replay(
    frags: &mut BTreeMap<u64, Fragment>,
    mut actions: Vec<pb::FragmentTreeMutation>,
    check: DeltaCheck,
) -> Result<()> {
    actions.sort_unstable_by_key(|tagged| tagged.action_sequence);
    if actions
        .windows(2)
        .any(|pair| pair[0].action_sequence == pair[1].action_sequence)
    {
        return Err(super::validation::corrupt(
            "Duplicate action sequence number on a fragment replay path",
        ));
    }
    // Upserted records are validated as `apply_action` decodes them.
    for tagged in &actions {
        super::validation::action_target(tagged.action.as_ref().ok_or_else(|| {
            super::validation::corrupt(format!(
                "Buffered action sequence number {} has no action",
                tagged.action_sequence
            ))
        })?)?;
    }
    actions.sort_by_key(|t| (action_key(t), t.action_sequence));
    for tagged in actions {
        let key = action_key(&tagged);
        let before = record_counts(frags.get(&key));
        if let Some(action) = tagged.action {
            apply_action(frags, action)?;
        }
        if check == DeltaCheck::Verify {
            let after = record_counts(frags.get(&key));
            let actual = (after.0 - before.0, after.1 - before.1, after.2 - before.2);
            let stored = (
                tagged.fragment_count_delta,
                tagged.total_rows_delta,
                tagged.visible_rows_delta,
            );
            if actual != stored {
                return Err(super::validation::corrupt(format!(
                    "Mutation action_sequence={} for fragment {key} stores fragment, row, and visible deltas {} {} {} but the record changed by {} {} {}",
                    tagged.action_sequence,
                    stored.0,
                    stored.1,
                    stored.2,
                    actual.0,
                    actual.1,
                    actual.2
                )));
            }
        }
    }
    Ok(())
}

/// Leaves have no slot for the metadata size hint yet, so the tree reports
/// every hint as unknown. A resident inventory, a replayed buffer and a leaf
/// then serve the same record.
pub(super) fn clear_metadata_size_hints(fragment: &mut Fragment) {
    for file in fragment.referenced_lance_files_mut() {
        file.file_metadata_size_bytes = None;
    }
}

fn apply_action(frags: &mut BTreeMap<u64, Fragment>, action: pb::FragmentAction) -> Result<()> {
    let Some(action) = action.action else {
        return Err(super::validation::corrupt(
            "Buffered action has no recognized variant",
        ));
    };
    match action {
        Action::UpsertFragment(f) => {
            let mut fragment = super::validation::fragment(f)?;
            clear_metadata_size_hints(&mut fragment);
            frags.insert(fragment.id, fragment);
        }
        Action::RemoveFragment(id) => {
            frags.remove(&id);
        }
        Action::AddDataFile(a) => {
            let mut file = crate::format::DataFile::try_from(
                a.file
                    .ok_or_else(|| Error::invalid_input("AddDataFile action missing file"))?,
            )?;
            // A leaf could not keep the hint, as for an upserted record.
            file.file_metadata_size_bytes = None;
            let fragment = frags.get_mut(&a.frag_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "AddDataFile action targets missing frag_id={}",
                    a.frag_id
                ))
            })?;
            fragment.files.push(file);
        }
        Action::RemoveDataFile(a) => {
            let fragment = frags.get_mut(&a.frag_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "RemoveDataFile action targets missing frag_id={}",
                    a.frag_id
                ))
            })?;
            fragment.files.retain(|f| f.path != a.path);
        }
        Action::ReplaceDataFile(a) => {
            if a.expected_path.is_empty() || a.path.is_empty() {
                return Err(Error::invalid_input(format!(
                    "ReplaceDataFile for frag_id={} is missing expected_path or path",
                    a.frag_id
                )));
            }
            let fragment = frags.get_mut(&a.frag_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "ReplaceDataFile action targets missing frag_id={}",
                    a.frag_id
                ))
            })?;
            let matched = fragment
                .files
                .iter_mut()
                .find(|file| file.path == a.expected_path)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "ReplaceDataFile for frag_id={} names data file {} which the fragment \
                         does not hold",
                        a.frag_id, a.expected_path
                    ))
                })?;
            matched.path = a.path.clone();
            matched.file_size_bytes = lance_io::utils::CachedFileSize::new(a.file_size_bytes);
            matched.base_id = a.base_id;
            // The action cannot carry the new file's hint, and the old hint
            // describes a different file.
            matched.file_metadata_size_bytes = None;
        }
        Action::AddDeletionFile(a) => {
            let deletion_file = a.deletion_file.ok_or_else(|| {
                Error::invalid_input(format!(
                    "AddDeletionFile action for frag_id={} is missing deletion_file",
                    a.frag_id
                ))
            })?;
            let fragment = frags.get_mut(&a.frag_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "AddDeletionFile action targets missing frag_id={}",
                    a.frag_id
                ))
            })?;
            let count = usize::try_from(deletion_file.num_deleted_rows)
                .map_err(|_| super::validation::corrupt("Deleted row count exceeds usize"))?;
            // Fragment::num_rows subtracts this count from the physical rows
            // unchecked, so an overcount must not reach the record.
            if let Some(rows) = fragment.physical_rows
                && count > rows
            {
                return Err(Error::invalid_input(format!(
                    "fragment {} has {count} deleted rows but only {rows} physical rows",
                    a.frag_id
                )));
            }
            let mut file = crate::format::DeletionFile::try_from(deletion_file)?;
            file.num_deleted_rows = Some(count);
            fragment.deletion_file = Some(file);
        }
        Action::ClearDeletionFile(a) => {
            let fragment = frags.get_mut(&a.frag_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "ClearDeletionFile action targets missing frag_id={}",
                    a.frag_id
                ))
            })?;
            fragment.deletion_file = None;
        }
    }
    Ok(())
}

/// Normalize validated current-state actions, preserving each fragment's effect.
///
/// Callers must validate actions before combining them and retain the originals
/// in transaction history. From `contiguous_from` onward, each fragment's history
/// must be contiguous within this buffer, with no intervening action in another
/// node. Earlier actions pass through unchanged.
///
/// The last whole-fragment reset establishes a known state, so its suffix can
/// be evaluated once. Otherwise, deletion-file assignments form an independent
/// last-writer-wins register, while file edits retain their order and first-path
/// matching semantics. In particular, a path-changing replacement chain cannot
/// be combined blindly: the second edit might match a different file slot.
///
/// Combined actions retain the last sequence number and the sum of aggregate
/// deltas. If a sum overflows, the run stays unchanged.
pub fn squash_buffer(
    mut buffer: Vec<pb::FragmentTreeMutation>,
    contiguous_from: u64,
) -> Vec<pb::FragmentTreeMutation> {
    // Distinct keys leave nothing to combine, and proving that costs far less
    // than grouping every action by key. The order matches the grouped path,
    // which sorts by action sequence with ties in key order.
    let mut keys: Vec<u64> = buffer
        .iter()
        .filter(|tagged| tagged.action_sequence >= contiguous_from)
        .map(action_key)
        .collect();
    keys.sort_unstable();
    if keys.windows(2).all(|pair| pair[0] != pair[1]) {
        buffer.sort_by_key(|tagged| (tagged.action_sequence, action_key(tagged)));
        return buffer;
    }
    let mut squashed = Vec::new();
    let mut runs: BTreeMap<u64, Vec<pb::FragmentTreeMutation>> = BTreeMap::new();
    for tagged in buffer {
        if tagged.action_sequence < contiguous_from {
            squashed.push(tagged);
        } else {
            runs.entry(action_key(&tagged)).or_default().push(tagged);
        }
    }
    for (key, mut run) in runs {
        run.sort_by_key(|tagged| tagged.action_sequence);
        if run.len() > 1 {
            run = squash_run(key, run);
        }
        squashed.extend(run);
    }
    squashed.sort_by_key(|tagged| tagged.action_sequence);
    squashed
}

fn combined(run: &[pb::FragmentTreeMutation], action: Action) -> Option<pb::FragmentTreeMutation> {
    Some(pb::FragmentTreeMutation {
        action_sequence: run.last()?.action_sequence,
        action: Some(pb::FragmentAction {
            action: Some(action),
        }),
        fragment_count_delta: run
            .iter()
            .try_fold(0i64, |sum, t| sum.checked_add(t.fragment_count_delta))?,
        total_rows_delta: run
            .iter()
            .try_fold(0i64, |sum, t| sum.checked_add(t.total_rows_delta))?,
        visible_rows_delta: run
            .iter()
            .try_fold(0i64, |sum, t| sum.checked_add(t.visible_rows_delta))?,
    })
}

fn squash_run(key: u64, run: Vec<pb::FragmentTreeMutation>) -> Vec<pb::FragmentTreeMutation> {
    if run
        .iter()
        .any(|t| t.action.as_ref().and_then(|a| a.action.as_ref()).is_none())
    {
        return run;
    }
    if let Some(reset) = run.iter().rposition(|t| {
        matches!(
            t.action.as_ref().and_then(|a| a.action.as_ref()),
            Some(Action::UpsertFragment(_)) | Some(Action::RemoveFragment(_))
        )
    }) {
        let mut state = BTreeMap::new();
        if apply_actions(&mut state, run[reset..].to_vec()).is_err() {
            return run;
        }
        let action = match state.remove(&key) {
            Some(fragment) => Action::UpsertFragment(pb::DataFragment::from(&fragment)),
            None => Action::RemoveFragment(key),
        };
        return combined(&run, action).map_or(run, |t| vec![t]);
    }

    // File edits leave the deletion register and fragment existence unchanged.
    // Its assignments therefore commute across those edits on validated input.
    let mut deletion: Vec<pb::FragmentTreeMutation> = Vec::new();
    let mut files: Vec<pb::FragmentTreeMutation> = Vec::with_capacity(run.len());
    for tagged in &run {
        let Some(next) = tagged.action.as_ref().and_then(|a| a.action.as_ref()) else {
            return run;
        };
        if matches!(
            next,
            Action::AddDeletionFile(_) | Action::ClearDeletionFile(_)
        ) {
            deletion.push(tagged.clone());
            continue;
        }
        let mut current = tagged.clone();
        while let Some(action) = files.last().and_then(|prev| {
            combine_file_pair(
                prev.action.as_ref()?.action.as_ref()?,
                current.action.as_ref()?.action.as_ref()?,
            )
        }) {
            let start = files.len() - 1;
            let pair = [files[start].clone(), current];
            let Some(merged) = combined(&pair, action) else {
                return run;
            };
            files.pop();
            current = merged;
        }
        files.push(current);
    }
    if let Some(last) = deletion.last() {
        let Some(action) = last.action.as_ref().and_then(|a| a.action.clone()) else {
            return run;
        };
        let Some(merged) = combined(&deletion, action) else {
            return run;
        };
        files.push(merged);
    }
    files.sort_by_key(|t| t.action_sequence);
    files
}

/// Only identities valid for ordered lists with arbitrary path aliases belong
/// here. Path uniqueness is not part of the storage action contract.
fn combine_file_pair(first: &Action, second: &Action) -> Option<Action> {
    match (first, second) {
        (Action::RemoveDataFile(a), Action::RemoveDataFile(b)) if a.path == b.path => {
            Some(Action::RemoveDataFile(b.clone()))
        }
        (Action::AddDataFile(a), Action::RemoveDataFile(b)) if a.file.as_ref()?.path == b.path => {
            // The remove still has to delete any same-path files in the base.
            Some(Action::RemoveDataFile(b.clone()))
        }
        (Action::ReplaceDataFile(a), Action::ReplaceDataFile(b))
            if a.expected_path == a.path && a.expected_path == b.expected_path =>
        {
            // The first edit did not rename the slot, so both find the same
            // first matching path, including when the base has duplicate paths.
            Some(Action::ReplaceDataFile(pb::ReplaceDataFile {
                frag_id: a.frag_id,
                expected_path: a.expected_path.clone(),
                path: b.path.clone(),
                file_size_bytes: b.file_size_bytes,
                base_id: b.base_id,
            }))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;
