// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::Error;
use lance_table::format::IndexMetadata;
use lance_table::system_index::{frag_reuse::lineage::TaggedLineage, is_system_index};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::Table;
use crate::commit::{CommitIndexHook, load_all_indices};

use super::decode_frag_reuse_ledger;

/// The committed segments whose coverage of `rewritten` a deferred rewrite
/// must withdraw instead of leaving behind: those that cover one of the
/// rewritten fragments, directly or through their retired provenance, and
/// whose type cannot translate stored addresses through the history in
/// batches.
///
/// The classification is the reader's own (`segment_supports_batch_remapping`),
/// so a segment the reader would translate keeps its provenance, except stale
/// segments (below):
/// - a segment the index hook reports stale (for a list-element segment, its
///   key fields no longer resolve in `dataset`'s schema) is listed before the
///   probe runs: nothing queries or rebuilds it, so keeping its provenance
///   would only pin the history;
/// - `Some(true)`: deferrable, not listed;
/// - `Some(false)`: listed, its coverage of the rewritten sources withdrawn;
/// - `None`: not listed. Details that cannot be interpreted are refused
///   elsewhere (the reader derives no coverage for such a segment and the
///   write gates refuse the shapes that would rely on one), and guessing a
///   withdrawal from unreadable metadata would delete coverage this build
///   cannot prove is unsound.
///
/// Returned in the stored listing's order so the result does not depend on
/// map iteration.
pub async fn undeferrable_rewrite_segments(
    dataset: &Table,
    rewritten: &RoaringBitmap,
    index_hook: Option<&dyn CommitIndexHook>,
) -> lance_core::Result<Vec<Uuid>> {
    if rewritten.is_empty() {
        return Ok(Vec::new());
    }
    let stored = load_all_indices(dataset, index_hook).await?;
    // A segment may cover `rewritten` only through a retired fragment an
    // earlier rewrite already moved into it, so the selector needs the same
    // history and liveness the withdrawal helper walks.
    let lineage = match stored
        .iter()
        .find(|index| lance_table::system_index::frag_reuse::metadata::is_tagged(index))
    {
        Some(entry) => {
            let ledger = decode_frag_reuse_ledger(dataset, entry).await?;
            Some(TaggedLineage::new(entry, Some(Arc::new(ledger))))
        }
        None => None,
    };
    let live: RoaringBitmap = dataset
        .manifest
        .fragments
        .iter()
        .filter_map(|fragment| u32::try_from(fragment.id).ok())
        .collect();
    undeferrable_segments_covering(
        dataset,
        stored.iter(),
        rewritten,
        lineage.as_ref(),
        &live,
        index_hook,
    )
    .await
}

/// [`undeferrable_rewrite_segments`] over a supplied segment list: the
/// segments of `segments` whose coverage of `fragments` cannot be translated
/// through the history. Used for segments the manifest does not list yet, a
/// `CreateIndex` rebased over a committed rewrite.
///
/// A segment is a candidate when it covers one of `fragments` directly, or
/// when it covers a retired fragment (outside `live`) whose rows the history
/// moved into one of `fragments`: that coverage addresses the rewritten
/// fragment just as much, through one more hop. The reach of a retired
/// source mirrors the withdrawal helper's conservative rule, so selection
/// and withdrawal agree on the same fragments: an unwalkable history
/// (`None` lineage, or `destinations_of` returning `None`) makes every
/// retired source a candidate rather than silently keeping coverage the
/// withdrawal would drop.
///
/// Only candidates pay the capability probe.
pub async fn undeferrable_segments_covering<'a>(
    dataset: &Table,
    segments: impl Iterator<Item = &'a IndexMetadata>,
    fragments: &RoaringBitmap,
    lineage: Option<&TaggedLineage>,
    live: &RoaringBitmap,
    index_hook: Option<&dyn CommitIndexHook>,
) -> lance_core::Result<Vec<Uuid>> {
    let mut undeferrable = Vec::new();
    let mut reaches: HashMap<u32, bool> = HashMap::new();
    for index in segments {
        if is_system_index(index) {
            continue;
        }
        // Unknown coverage is not a candidate, as in every earlier revision
        // of this selector: the established reader rules already refuse such
        // a segment (no derived coverage, scan fallback), so there is nothing
        // a withdrawal could protect, and the selector never guesses which
        // fragments an absent bitmap would have named.
        let Some(bitmap) = index.fragment_bitmap.as_ref() else {
            continue;
        };
        let covers_directly = !bitmap.is_disjoint(fragments);
        let covers_through_history = !covers_directly
            && (bitmap - live).iter().any(|source| {
                // Same-name segments routinely share retired sources, and the
                // walk is transitive; memoize it for this call.
                *reaches.entry(source).or_insert_with(|| {
                    match lineage.and_then(|lineage| lineage.destinations_of(source)) {
                        Some(destinations) => destinations
                            .iter()
                            .any(|destination| fragments.contains(*destination)),
                        None => true,
                    }
                })
            });
        if !covers_directly && !covers_through_history {
            continue;
        }
        // TODO: Support deferred remapping for all index types.
        // Only a CONFIRMED undeferrable verdict withdraws coverage. A probe
        // that cannot reach the segment's format marker is not a verdict: the
        // commit proceeds without selecting the segment, so no withdrawal is
        // persisted from a transient fault and nothing is cached. The reader
        // refuses such a segment's coverage meanwhile, and a later commit
        // withdraws once the probe answers.
        let hook = index_hook.ok_or_else(|| {
            Error::not_supported(format!(
                "Cannot check fragment reuse remapping support for index {} (segment {}) without a commit index hook",
                index.name, index.uuid,
            ))
        })?;
        // A stale segment (for a list-element segment, a key was dropped or
        // retyped in the schema this rewrite builds on) is never queried or
        // rebuilt, so a deferred rewrite must withdraw its coverage rather than
        // leave it pinning the history. This is deliberately not part of the
        // reader's probe: `remap_column_index` can still catch such a segment
        // up. An undecodable one is unknown, not a verdict.
        match hook.is_stale_segment(dataset, index) {
            Ok(true) => {
                undeferrable.push(index.uuid);
                continue;
            }
            Ok(false) => {}
            Err(_) => continue,
        }
        match hook.supports_batch_remapping(dataset, index).await {
            Ok(Some(false)) => undeferrable.push(index.uuid),
            Ok(_) => {}
            Err(error) => log::warn!(
                "index {} (segment {}): could not read its format marker ({error}); \
                 not withdrawing its coverage in this commit",
                index.name,
                index.uuid,
            ),
        }
    }
    Ok(undeferrable)
}
