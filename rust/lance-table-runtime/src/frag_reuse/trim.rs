// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::Error;
use lance_table::format::IndexMetadata;
use lance_table::io::manifest::read_manifest_indexes;
use lance_table::system_index::frag_reuse::{
    FRAG_REUSE_INDEX_NAME,
    metadata::is_tagged,
    retention::{TrimRecord, compute_tagged_retention, split_trim_elements},
};
use lance_table::transaction::{Operation, Transaction};
use roaring::RoaringBitmap;

use crate::Table;
use crate::commit::{CommitIndexHook, index::index_usability};

use super::{
    build_tagged_frag_reuse_entry, decode_frag_reuse_ledger_from_content,
    load_raw_frag_reuse_content,
};

/// Conflict-source marker for a trim whose rebase derived nothing left to
/// trim: a concurrent commit already satisfied it, so instead of writing an
/// empty no-op manifest version the commit attempt aborts with a retryable
/// conflict carrying this message, and the FRI cleanup treats
/// exactly that conflict as success.
pub const TAGGED_TRIM_REBASED_TO_NOOP: &str =
    "the tagged trim rebased to nothing to trim; a concurrent commit already satisfied it";

/// Conflict-source marker for a superseded-segment prune whose rebase derived
/// nothing left to remove: a concurrent commit changed the coverage the
/// original removal set relied on (or already removed the segments), so
/// instead of writing an empty no-op manifest version the commit attempt
/// aborts with a retryable conflict carrying this message, and
/// maintenance treats exactly that conflict as success.
pub const SUPERSEDED_PRUNE_REBASED_TO_NOOP: &str = "the superseded-segment prune rebased to nothing to remove; \
     a concurrent commit changed the coverage it relied on";

/// Transaction property marking a removal-only `CreateIndex` as a
/// superseded-segment prune.
///
/// The shape alone (removals, no creations) is ambiguous: `drop_index`
/// commits exactly the same operation, and re-deriving a user's drop as a
/// prune would resurrect indexes they asked to remove. The explicit property
/// travels with the transaction through the commit loop, so the conflict
/// resolver's `finish_create_index` can recognize the prune and re-derive its
/// removal set against the current manifest whenever the attempt builds on a
/// version newer than the one it read. Without that, a rebase over a
/// non-conflicting concurrent commit (e.g. a remap swap narrowing a sibling's
/// coverage) would re-commit the stale removal set and silently drop index
/// coverage.
pub const SUPERSEDED_PRUNE_PROPERTY: &str = "__lance.frag_reuse.superseded_prune";

/// Whether a transaction is a superseded-segment prune: marked with
/// [`SUPERSEDED_PRUNE_PROPERTY`] and shaped as a pure removal.
pub fn is_superseded_prune_transaction(transaction: &Transaction) -> bool {
    let has_marker = transaction
        .transaction_properties
        .as_ref()
        .is_some_and(|props| props.contains_key(SUPERSEDED_PRUNE_PROPERTY));
    has_marker
        && matches!(
            &transaction.operation,
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } if new_indices.is_empty() && !removed_indices.is_empty()
        )
}

/// The outcome of deriving a tagged trim against the current manifest.
#[derive(Debug)]
pub enum TaggedTrimOutcome {
    /// Every record is still needed, or there is no tagged entry to trim.
    NothingToTrim,
    /// Replace the current entry with the filtered history.
    Replace {
        new_entry: Box<IndexMetadata>,
        current_entry: Box<IndexMetadata>,
    },
    /// Everything trimmed away: delete the entry outright. The manifest
    /// feature flags stay set (they are sticky).
    Delete { current_entry: Box<IndexMetadata> },
}

/// Whether a `CreateIndex` payload has the tagged-trim shape: exactly the
/// tagged FRI entry replaced, or removed outright. A v0 trim (v0 entry
/// replacing a v0 entry) deliberately does not match.
pub fn is_tagged_trim_operation(
    new_indices: &[IndexMetadata],
    removed_indices: &[IndexMetadata],
) -> bool {
    let [removed] = removed_indices else {
        return false;
    };
    if removed.name != FRAG_REUSE_INDEX_NAME {
        return false;
    }
    match new_indices {
        [] => is_tagged(removed),
        [entry] => is_tagged(entry),
        _ => false,
    }
}

/// Derive the trimmed tagged entry from the CURRENT manifest state.
///
/// Zero payload IO: retention is computed from the entry's digests, the
/// stored index metadata and the manifest's live fragments alone. Row-map
/// files released by trimming a stable-partition transition are collected
/// separately by dataset cleanup.
pub async fn derive_tagged_trim(
    dataset: &Table,
    index_hook: Option<&dyn CommitIndexHook>,
) -> lance_core::Result<TaggedTrimOutcome> {
    let indices = read_manifest_indexes(
        &dataset.state.object_store,
        &dataset.state.manifest_location,
        &dataset.manifest,
    )
    .await?;
    let Some(entry) = indices
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .cloned()
    else {
        return Ok(TaggedTrimOutcome::NothingToTrim);
    };
    if entry.index_version == 0 {
        // Reachable only through a rebase whose entry was concurrently
        // replaced by a v0 one; nothing tagged remains to trim.
        return Ok(TaggedTrimOutcome::NothingToTrim);
    }

    let content = load_raw_frag_reuse_content(dataset, &entry).await?;
    // Full ledger validation (lineage order, digest conservation, mapping
    // presence) plus unsupported-transition detection before interpreting
    // anything.
    let ledger = decode_frag_reuse_ledger_from_content(entry.index_version, &content).await?;
    if ledger.has_unsupported_transitions() {
        return Err(Error::not_supported(
            "the tagged FRI history carries transitions this client cannot interpret; \
             upgrade to a newer version of Lance before trimming",
        ));
    }
    let elements = split_trim_elements(&content)?;
    // The live fragments of the same snapshot `indices` was read from; the
    // commit-time rebase calls this again with the rebased dataset.
    let retained = compute_tagged_retention(
        &elements,
        &indices,
        dataset.state.fragment_bitmap.as_ref(),
        index_usability(index_hook),
    );

    if retained.iter().all(|kept| *kept) {
        return Ok(TaggedTrimOutcome::NothingToTrim);
    }
    if retained.iter().all(|kept| !*kept) {
        return Ok(TaggedTrimOutcome::Delete {
            current_entry: Box::new(entry),
        });
    }

    let mut new_content = Vec::new();
    let mut fragment_bitmap = RoaringBitmap::new();
    for (element, kept) in elements.iter().zip(retained.iter()) {
        if !kept {
            continue;
        }
        new_content.extend_from_slice(&element.raw);
        match &element.record {
            TrimRecord::LegacyVersion(version) => {
                fragment_bitmap.extend(version.old_frag_ids().iter().map(|&id| id as u32));
                fragment_bitmap.extend(version.new_frag_ids().iter().map(|&id| id as u32));
            }
            TrimRecord::Transition(transition) => {
                for digest in transition
                    .sources
                    .iter()
                    .chain(transition.destinations.iter())
                {
                    fragment_bitmap.insert(digest.id as u32);
                }
            }
        }
    }
    // The filtered history must itself decode as a well-formed ledger.
    decode_frag_reuse_ledger_from_content(entry.index_version, &new_content).await?;
    let new_entry = build_tagged_frag_reuse_entry(dataset, new_content, fragment_bitmap).await?;
    Ok(TaggedTrimOutcome::Replace {
        new_entry: Box::new(new_entry),
        current_entry: Box::new(entry),
    })
}
