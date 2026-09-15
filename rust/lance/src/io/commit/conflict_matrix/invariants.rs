// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Properties every committed manifest must hold, whatever it was rebased over.
//!
//! These are deliberately stated over *observable* dataset state — the manifest
//! a reader loads and the rows a scan returns — rather than over the conflict
//! resolver's internals. A rewrite of conflict detection may reshape
//! `TransactionRebase` freely; it may not make one of these false.
//!
//! The battery runs after every landing in the matrix, so a bug that only shows
//! up in one cell is caught by the same code that covers the other hundred and
//! thirty-nine.
//!
//! Each one earns its place by being able to fail. `Manifest::max_field_id` is
//! deliberately *not* asserted against the schema: it is defined as the maximum
//! over the schema and the fragments, so "every schema field id is within the
//! watermark" holds by construction. The observable half of that property — a
//! data file the schema can no longer reach — is [`no_orphaned_data_file`].

use std::collections::HashSet;

use lance_table::format::Fragment;

use super::scenarios::{Scenario, row_counts};
use crate::Dataset;

/// Check every invariant against a landing, panicking with `context` on the
/// first failure.
///
/// `before` is the dataset the losing transaction read; `after` is what it
/// produced. `ours` names the operation that just landed, which decides whether
/// the incremental invariants apply at all.
pub(super) async fn check_all(before: &Dataset, after: &Dataset, ours: Scenario, context: &str) {
    no_orphaned_data_file(after, context);
    overlays_reference_live_fragments(after, context);
    tombstones_stay_tombstoned(before, after, context);
    if !ours.replaces_state() {
        rows_are_not_resurrected(before, after, context);
        overlays_are_not_dropped(before, after, context);
    }
    manifest_round_trips(after, context);
    dataset_scans(after, context).await;
}

/// Every data file must contribute at least one field the schema still has.
///
/// A file none of whose fields are in the schema is unreachable by any reader
/// and, worse, is evidence that a post-image built before a concurrent schema
/// change was installed over it — the file was pruned and then reinstated.
///
/// Note the weaker form: a file may legitimately carry a field the schema has
/// dropped, because a projection only prunes a file when *all* of its fields go
/// away. Asserting the stronger "every field id appears in the schema" would
/// fire on an ordinary single-fragment projection with no conflict at all.
fn no_orphaned_data_file(after: &Dataset, context: &str) {
    let live: HashSet<i32> = after
        .manifest
        .schema
        .fields_pre_order()
        .map(|field| field.id)
        .collect();
    for fragment in after.fragments().iter() {
        for file in fragment.files.iter() {
            assert!(
                file.fields.iter().any(|id| live.contains(id)),
                "{context}: fragment {} carries data file {:?} whose fields {:?} are all absent \
                 from the schema {live:?} — a pruned file was reinstated",
                fragment.id,
                file.path,
                file.fields,
            );
        }
    }
}

/// An overlay must hang off a fragment that still exists, and supply fields the
/// schema still has. A dangling overlay is unreadable.
fn overlays_reference_live_fragments(after: &Dataset, context: &str) {
    let live_fields: HashSet<i32> = after
        .manifest
        .schema
        .fields_pre_order()
        .map(|field| field.id)
        .collect();
    for fragment in after.fragments().iter() {
        for overlay in fragment.overlays.iter() {
            assert!(
                overlay
                    .data_file
                    .fields
                    .iter()
                    .any(|id| live_fields.contains(id)),
                "{context}: fragment {} has an overlay supplying fields {:?}, none of which are \
                 in the schema {live_fields:?}",
                fragment.id,
                overlay.data_file.fields,
            );
        }
    }
}

/// A field tombstoned to `-2` must stay tombstoned: its data is gone, so
/// reinstating the id would resurrect a column the writer proved was dropped.
///
/// Legacy operations do not tombstone — only the action vocabulary does — so
/// this is vacuous today and is here to fail loudly the moment that changes.
fn tombstones_stay_tombstoned(before: &Dataset, after: &Dataset, context: &str) {
    const TOMBSTONE: i32 = -2;
    let tombstoned: HashSet<String> = before
        .manifest
        .schema
        .fields_pre_order()
        .filter(|field| field.id == TOMBSTONE)
        .map(|field| field.name.clone())
        .collect();
    if tombstoned.is_empty() {
        return;
    }
    for field in after.manifest.schema.fields_pre_order() {
        if tombstoned.contains(&field.name) {
            assert_eq!(
                field.id, TOMBSTONE,
                "{context}: field {} was tombstoned and came back with id {}",
                field.name, field.id,
            );
        }
    }
}

/// A fragment that survives a commit untouched must not gain rows.
///
/// Deletions are monotonic within a fragment: nothing short of a rewrite puts a
/// deleted row back. A fragment whose logical count went *up* means a stale
/// post-image overwrote a deletion file that had moved on.
fn rows_are_not_resurrected(before: &Dataset, after: &Dataset, context: &str) {
    let rewritten = rewritten_fragments(before, after);
    let was = row_counts(before);
    let now = row_counts(after);
    for (id, count) in now {
        if rewritten.contains(&id) {
            continue;
        }
        let Some(previous) = was.get(&id) else {
            // A newly minted fragment has nothing to compare against.
            continue;
        };
        assert!(
            count <= *previous,
            "{context}: fragment {id} went from {previous} live rows to {count} without being \
             rewritten — a concurrent delete was reverted",
        );
    }
}

/// A fragment that survives a commit untouched must not lose overlays.
///
/// An overlay supplies the current value of a cell. Dropping one silently
/// reverts that cell to its base value, which a reader cannot distinguish from
/// the write never having happened.
fn overlays_are_not_dropped(before: &Dataset, after: &Dataset, context: &str) {
    let rewritten = rewritten_fragments(before, after);
    let was = overlay_counts(before);
    for fragment in after.fragments().iter() {
        if rewritten.contains(&fragment.id) {
            continue;
        }
        let Some(previous) = was.get(&fragment.id) else {
            continue;
        };
        assert!(
            fragment.overlays.len() >= *previous,
            "{context}: fragment {} went from {previous} overlays to {} without being rewritten \
             — a concurrent overlay was dropped",
            fragment.id,
            fragment.overlays.len(),
        );
    }
}

/// Fragments whose data files changed identity, which is how a rewrite,
/// replacement or column update shows up. Those legitimately reset per-fragment
/// state, so the monotonic invariants skip them.
fn rewritten_fragments(before: &Dataset, after: &Dataset) -> HashSet<u64> {
    let paths = |fragment: &Fragment| {
        fragment
            .files
            .iter()
            .map(|file| file.path.clone())
            .collect::<HashSet<_>>()
    };
    let was: std::collections::HashMap<u64, HashSet<String>> = before
        .fragments()
        .iter()
        .map(|fragment| (fragment.id, paths(fragment)))
        .collect();
    after
        .fragments()
        .iter()
        .filter(|fragment| {
            was.get(&fragment.id)
                .is_none_or(|previous| *previous != paths(fragment))
        })
        .map(|fragment| fragment.id)
        .collect()
}

fn overlay_counts(dataset: &Dataset) -> std::collections::HashMap<u64, usize> {
    dataset
        .fragments()
        .iter()
        .map(|fragment| (fragment.id, fragment.overlays.len()))
        .collect()
}

/// The manifest must survive the protobuf encoding it is about to be written
/// in. A manifest that only exists in memory has not really been committed.
fn manifest_round_trips(after: &Dataset, context: &str) {
    let encoded = lance_table::format::pb::Manifest::from(after.manifest.as_ref());
    let decoded = lance_table::format::Manifest::try_from(encoded)
        .unwrap_or_else(|e| panic!("{context}: manifest failed to round-trip through pb: {e}"));
    assert_eq!(
        decoded.version, after.manifest.version,
        "{context}: manifest version changed across a pb round-trip",
    );
    assert_eq!(
        decoded.fragments.len(),
        after.manifest.fragments.len(),
        "{context}: fragment count changed across a pb round-trip",
    );
}

/// The whole point: the dataset still reads. Every invariant above is a
/// structural proxy for this one.
async fn dataset_scans(after: &Dataset, context: &str) {
    let batch = after
        .scan()
        .try_into_batch()
        .await
        .unwrap_or_else(|e| panic!("{context}: the committed dataset does not scan: {e}"));
    let expected: u64 = row_counts(after).values().sum();
    assert_eq!(
        batch.num_rows() as u64,
        expected,
        "{context}: scan returned {} rows but the manifest accounts for {expected}",
        batch.num_rows(),
    );
}
