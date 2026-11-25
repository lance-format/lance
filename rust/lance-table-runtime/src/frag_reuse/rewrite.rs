// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};

use lance_core::Error;
use lance_table::format::pb::fragment_reuse_index_details::{self as pb_fri, InlineContent};
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_table::system_index::frag_reuse::gate::{BareRewriteRefusal, covered_bare_rewrite};
use lance_table::transaction::RewriteGroup;
use prost::Message;
use roaring::RoaringBitmap;

use crate::Table;
use crate::commit::{CommitIndexHook, load_all_indices};

use super::deletions::{current_deleted_rows, validate_folded_deletions};
use super::storage::{
    build_tagged_frag_reuse_entry, decode_records, encode_length_delimited_field,
    load_raw_frag_reuse_content,
};

/// What identifies a transition: the fragments it consumed and the ones it
/// produced, in order. A source fragment is consumed by one transition only
/// and destinations are freshly reserved, so two transitions with the same
/// identity are the same record, and must then carry the same content.
fn transition_identity(transition: &pb_fri::Transition) -> (Vec<u64>, Vec<u64>) {
    (
        transition.sources.iter().map(|digest| digest.id).collect(),
        transition
            .destinations
            .iter()
            .map(|digest| digest.id)
            .collect(),
    )
}

/// The records a rewrite's entry adds relative to the entry at its read
/// version.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct AddedRecords {
    pub transitions: Vec<pb_fri::Transition>,
    pub legacy_versions: Vec<pb_fri::Version>,
}

/// Work out what `proposed` (a rewrite's `frag_reuse_index`) adds relative
/// to `base` (the entry at the rewrite's read version), both as records.
///
/// A rewrite only appends history. A record `base` holds must appear in
/// `proposed` with the same content: one that is missing is a removal the
/// rewrite is not allowed to make (history is retired through the reuse
/// index cleanup, which proves nothing still depends on it), and one whose
/// content differs would rewrite a record other indices may translate
/// through. Transitions are identified by their source and destination
/// fragments, legacy versions by their dataset version.
pub fn records_added_since(
    base: &InlineContent,
    proposed: &InlineContent,
) -> lance_core::Result<AddedRecords> {
    let mut added = AddedRecords::default();

    let proposed_transitions: HashMap<(Vec<u64>, Vec<u64>), &pb_fri::Transition> = proposed
        .transitions
        .iter()
        .map(|transition| (transition_identity(transition), transition))
        .collect();
    if proposed_transitions.len() != proposed.transitions.len() {
        return Err(Error::invalid_input(
            "the fragment reuse entry records the same transition more than once",
        ));
    }
    let mut base_transitions: HashMap<(Vec<u64>, Vec<u64>), &pb_fri::Transition> = HashMap::new();
    for transition in &base.transitions {
        let identity = transition_identity(transition);
        match proposed_transitions.get(&identity) {
            None => {
                return Err(Error::invalid_input(format!(
                    "the fragment reuse entry drops the recorded transition from fragments \
                     {:?}; a rewrite only appends history, retire transitions through the \
                     fragment reuse index cleanup instead",
                    identity.0
                )));
            }
            Some(proposed) if *proposed != transition => {
                return Err(Error::invalid_input(format!(
                    "the fragment reuse entry changes the recorded transition from fragments \
                     {:?}; a recorded transition cannot be modified",
                    identity.0
                )));
            }
            Some(_) => {}
        }
        base_transitions.insert(identity, transition);
    }
    for transition in &proposed.transitions {
        if !base_transitions.contains_key(&transition_identity(transition)) {
            added.transitions.push(transition.clone());
        }
    }

    let proposed_versions: HashMap<u64, &pb_fri::Version> = proposed
        .legacy_versions
        .iter()
        .map(|version| (version.dataset_version, version))
        .collect();
    if proposed_versions.len() != proposed.legacy_versions.len() {
        return Err(Error::invalid_input(
            "the fragment reuse entry records the same compaction version more than once",
        ));
    }
    let mut base_versions: HashSet<u64> = HashSet::new();
    for version in &base.legacy_versions {
        match proposed_versions.get(&version.dataset_version) {
            None => {
                return Err(Error::invalid_input(format!(
                    "the fragment reuse entry drops the recorded compaction version {}; a \
                     rewrite only appends history, retire versions through the fragment \
                     reuse index cleanup instead",
                    version.dataset_version
                )));
            }
            Some(proposed) if *proposed != version => {
                return Err(Error::invalid_input(format!(
                    "the fragment reuse entry changes the recorded compaction version {}; a \
                     recorded version cannot be modified",
                    version.dataset_version
                )));
            }
            Some(_) => {}
        }
        base_versions.insert(version.dataset_version);
    }
    for version in &proposed.legacy_versions {
        if !base_versions.contains(&version.dataset_version) {
            added.legacy_versions.push(version.clone());
        }
    }
    Ok(added)
}

/// A legacy compaction version as the tagged history records it: one
/// ordered-compaction transition per group, informationally equivalent
/// (same digest ordering, same `changed_row_addrs` semantics).
pub fn legacy_version_transitions(version: &pb_fri::Version) -> Vec<pb_fri::Transition> {
    version
        .groups
        .iter()
        .map(|group| pb_fri::Transition {
            sources: group.old_fragments.clone(),
            destinations: group.new_fragments.clone(),
            mapping: Some(pb_fri::transition::Mapping::OrderedCompaction(
                pb_fri::OrderedCompaction {
                    changed_row_addrs: group.changed_row_addrs.clone(),
                },
            )),
        })
        .collect()
}

/// Refuse a bare group of a transition-carrying rewrite by the bare-rewrite
/// rule (`gate::covered_bare_rewrite`).
fn refuse_covered_bare_group(
    group: &RewriteGroup,
    indices: &[IndexMetadata],
) -> lance_core::Result<()> {
    let rewritten: RoaringBitmap = group
        .old_fragments
        .iter()
        .filter_map(|fragment| u32::try_from(fragment.id).ok())
        .collect();
    let fragments = || {
        group
            .old_fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>()
    };
    match covered_bare_rewrite(&rewritten, indices) {
        None => Ok(()),
        Some(BareRewriteRefusal::Covered { index, overlap }) => Err(Error::invalid_input(format!(
            "rewrite group of fragments {:?} carries no fragment reuse transition but index \
             {index} covers fragments {:?} of it; record a transition for the group or rebuild \
             the index before the rewrite",
            fragments(),
            overlap.iter().collect::<Vec<_>>(),
        ))),
        Some(BareRewriteRefusal::UnknownCoverage { index }) => Err(Error::invalid_input(format!(
            "rewrite group of fragments {:?} carries no fragment reuse transition and index \
             {index} has no fragment bitmap, so its coverage is unknown; record a transition \
             for the group or rebuild the index before the rewrite",
            fragments(),
        ))),
    }
}

/// Assemble the tagged FRI entry a rewrite commits, and return it with the
/// `dataset_version` of the entry it appended onto.
///
/// `transitions` are the records the rewrite adds (what its
/// `frag_reuse_index` holds beyond the entry at its read version, see
/// `records_added_since`). The current entry's content bytes are carried
/// over verbatim (a v0 entry is lifted to index_version 1 by
/// reinterpretation, not re-encoding) and each new transition is appended as
/// another `InlineContent.transitions` element. Before anything is spilled
/// or committed, the binding between the rewrite groups and the transitions
/// is validated (sources and destinations must match the groups' old and new
/// fragments one to one, in order), row counts must be conserved, no added
/// transition may already be recorded, and the whole assembled content must
/// decode as a valid ledger.
pub async fn build_frag_reuse_rewrite_entry(
    dataset: &Table,
    transitions: &[pb_fri::Transition],
    groups: &[RewriteGroup],
    index_hook: Option<&dyn CommitIndexHook>,
) -> lance_core::Result<(IndexMetadata, Option<u64>)> {
    // Deferred compaction records v0 entries on stable-row-id tables, so only a
    // hand-built tagged rewrite committed directly reaches this.
    if dataset.manifest.uses_stable_row_ids() {
        return Err(Error::not_supported(
            "Tagged fragment reuse histories are address-based and excluded on \
             stable-row-id datasets; this rewrite cannot carry transition intent here",
        ));
    }

    if transitions.is_empty() {
        return Err(Error::invalid_input(
            "a fragment-reuse rewrite carries no transitions",
        ));
    }

    let stored = load_all_indices(dataset, index_hook).await?;

    // Bind the covered rewrite groups to the transitions, one to one and in
    // order. A group is covered when its old fragments appear among the
    // transitions' sources; a group straddling covered and uncovered sources
    // is rejected (see `ordered_rewrite_groups`). A bare group follows the
    // bare-rewrite rule against the index list of this attempt.
    let source_ids: HashSet<u64> = transitions
        .iter()
        .flat_map(|transition| transition.sources.iter().map(|source| source.id))
        .collect();
    let mut covered_groups = Vec::with_capacity(transitions.len());
    for group in groups {
        let covered = group
            .old_fragments
            .iter()
            .filter(|frag| source_ids.contains(&frag.id))
            .count();
        if covered == 0 {
            refuse_covered_bare_group(group, &stored)?;
            continue;
        }
        if covered != group.old_fragments.len() {
            return Err(Error::invalid_input(
                "a rewrite group mixes transition-covered and order-preserving source fragments",
            ));
        }
        covered_groups.push(group);
    }
    if covered_groups.len() != transitions.len() {
        return Err(Error::invalid_input(format!(
            "the fragment-reuse rewrite lists {} transitions but {} rewrite groups are \
             covered by their sources",
            transitions.len(),
            covered_groups.len()
        )));
    }
    // Destination ids must be finalized before assembly: the commit path's
    // `fragments_with_ids` treats id 0 as unassigned and renumbers it, which
    // would strand the recorded destination id (and, with a live fragment 0,
    // translate into unrelated rows). The flow reserves fresh ids through
    // ReserveFragments; enforce that invariant here instead of trusting the
    // caller: no id 0, and no id that is already live in the manifest (the
    // transition's sources are live and are covered by the same rule).
    let live_fragments: HashSet<u64> = dataset.fragments().iter().map(|frag| frag.id).collect();
    let manifest_by_id: HashMap<u64, &Fragment> = dataset
        .fragments()
        .iter()
        .map(|frag| (frag.id, frag))
        .collect();

    for (group, transition) in covered_groups.iter().zip(transitions.iter()) {
        if group.old_fragments.len() != transition.sources.len() {
            return Err(Error::invalid_input(format!(
                "a transition lists {} sources but its rewrite group holds {} old fragments",
                transition.sources.len(),
                group.old_fragments.len()
            )));
        }
        // Regime detection: the per-source delta between the scan-time
        // digest and the CURRENT manifest. All deltas zero -> regime A,
        // exactly the historical validation (metadata-only here, except the
        // rare read of a deletion vector whose count was never
        // materialized). Any delta positive -> the folding regime: a concurrent delete landed on a
        // source after the job snapshotted it, and the job answered by
        // folding those rows into destination deletion vectors instead of
        // recomputing; the fold is then validated row by row against the
        // transition's own row map (see `validate_folded_deletions`). A
        // source missing from the manifest contributes no delta: that
        // commit fails later exactly as before (double consumption or
        // fragment liveness).
        let mut source_deltas: Vec<u64> = Vec::with_capacity(transition.sources.len());
        for digest in transition.sources.iter() {
            let current = match manifest_by_id.get(&digest.id) {
                None => digest.num_deleted_rows,
                Some(frag) => current_deleted_rows(dataset, frag).await?,
            };
            if current < digest.num_deleted_rows {
                return Err(Error::invalid_input(format!(
                    "source fragment {} records {current} deleted rows in the manifest, below \
                     the {} its scan-time digest carries; a deletion vector cannot shrink",
                    digest.id, digest.num_deleted_rows
                )));
            }
            source_deltas.push(current - digest.num_deleted_rows);
        }
        let folding = source_deltas.iter().any(|&delta| delta > 0);

        for (source_index, (frag, digest)) in group
            .old_fragments
            .iter()
            .zip(transition.sources.iter())
            .enumerate()
        {
            // Materializing a data overlay breaks the reuse premise that a
            // rewrite moves addresses, never values: the destination holds
            // the overlaid values physically (and carries no overlay for
            // the runtime staleness guard to see), while an index built
            // before the overlay keeps the source in its bitmap as
            // provenance and would serve the destination through
            // translation. v0 handles this by dropping the DESTINATION ids
            // from stale bitmaps, which is a no-op here because tagged
            // provenance never contains them. Until provenance-side
            // invalidation exists, refuse to record a transition over
            // overlaid sources.
            if !frag.overlays.is_empty() {
                return Err(Error::not_supported(format!(
                    "source fragment {} carries data overlay files; a rewrite recording \
                     fragment reuse transitions would materialize the overlaid values while \
                     indices keep translated coverage over the old addresses. Compact the \
                     overlays away or rebuild the covering indices eagerly before this \
                     rewrite",
                    frag.id
                )));
            }
            let physical_rows = frag.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "source fragment {} has no physical row count",
                    frag.id
                ))
            })? as u64;
            let num_deleted_rows = if folding {
                // One authoritative accessor in the folding regime, so an
                // unmaterialized count cannot be classified one way and
                // validated another.
                current_deleted_rows(dataset, frag).await?
            } else {
                frag.deletion_file
                    .as_ref()
                    .and_then(|deletion| deletion.num_deleted_rows)
                    .unwrap_or(0) as u64
            };
            if !folding {
                // Regime A: exactly the historical binding.
                if digest.id != frag.id
                    || digest.physical_rows != physical_rows
                    || digest.num_deleted_rows != num_deleted_rows
                {
                    return Err(Error::invalid_input(format!(
                        "transition source digest {:?} does not match old fragment {} \
                         ({physical_rows} physical rows, {num_deleted_rows} deleted)",
                        digest, frag.id
                    )));
                }
            } else {
                // Regime B: the digest keeps the scan-time snapshot; the
                // group must carry the source's CURRENT metadata.
                if digest.id != frag.id || digest.physical_rows != physical_rows {
                    return Err(Error::invalid_input(format!(
                        "transition source digest {:?} does not match old fragment {} \
                         ({physical_rows} physical rows)",
                        digest, frag.id
                    )));
                }
                let current = digest.num_deleted_rows + source_deltas[source_index];
                if num_deleted_rows != current {
                    return Err(Error::invalid_input(format!(
                        "deletion folding requires the rewrite group to carry the source's \
                         current metadata: fragment {} shows {num_deleted_rows} deleted rows \
                         but the manifest records {current}",
                        frag.id
                    )));
                }
            }
        }
        if group.new_fragments.len() != transition.destinations.len() {
            return Err(Error::invalid_input(format!(
                "a transition lists {} destinations but its rewrite group holds {} new fragments",
                transition.destinations.len(),
                group.new_fragments.len()
            )));
        }
        for (frag, digest) in group
            .new_fragments
            .iter()
            .zip(transition.destinations.iter())
        {
            if frag.id == 0 {
                return Err(Error::invalid_input(
                    "destination fragment id 0 is unassigned (the commit path renumbers it); \
                     reserve ids with ReserveFragments and assign them before the rewrite is \
                     assembled",
                ));
            }
            if live_fragments.contains(&frag.id) {
                return Err(Error::invalid_input(format!(
                    "destination fragment id {} is already live in the dataset; a rewrite \
                     destination must use a freshly reserved id",
                    frag.id
                )));
            }
            // The digest claiming zero deletions is not enough: check the
            // fragment itself, or a destination carrying a deletion file
            // would commit a digest that undercounts its physical rows'
            // liveness and later fail (or falsely pass) translation. Under
            // the folding regime destinations MAY carry deletion vectors --
            // they hold exactly the delta rows, validated position by
            // position below.
            if !folding && frag.deletion_file.is_some() {
                return Err(Error::invalid_input(format!(
                    "destination fragment {} carries a deletion file; rewrite destinations                      must be written without deletions",
                    frag.id
                )));
            }
            let physical_rows = frag.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "destination fragment {} has no physical row count",
                    frag.id
                ))
            })? as u64;
            if digest.id != frag.id
                || digest.physical_rows != physical_rows
                || digest.num_deleted_rows != 0
            {
                return Err(Error::invalid_input(format!(
                    "transition destination digest {:?} does not match new fragment {} \
                     ({physical_rows} physical rows)",
                    digest, frag.id
                )));
            }
        }
        // Conservation: every live source row lands in exactly one
        // destination. The digests were just bound to the actual fragments,
        // so this checks the fragments themselves.
        let live_source_rows: u64 = transition
            .sources
            .iter()
            .map(|digest| digest.physical_rows.saturating_sub(digest.num_deleted_rows))
            .sum();
        let destination_rows: u64 = transition
            .destinations
            .iter()
            .map(|digest| digest.physical_rows)
            .sum();
        if live_source_rows != destination_rows {
            return Err(Error::invalid_input(format!(
                "a transition does not conserve rows: {live_source_rows} live source rows, \
                 {destination_rows} destination rows"
            )));
        }
        if folding {
            validate_folded_deletions(dataset, group, transition, &manifest_by_id, &source_deltas)
                .await?;
        }
        // TODO(row-map totals): also validate the transition's row-map label
        // totals against the destination digests by tail-reading the map
        // file's counts buffer (RowMapReader keeps per-destination totals);
        // today that costs one object-store read per transition, so the
        // ledger's digest conservation stands in for it at commit time.
    }

    // Carry the current entry's content bytes over verbatim.
    let existing = stored.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME);
    let (mut content, base_bitmap, base_entry_version) = match existing {
        None => (Vec::new(), RoaringBitmap::new(), None),
        Some(entry) => {
            if !matches!(entry.index_version, 0 | 1) {
                return Err(Error::not_supported(format!(
                    "Cannot append a stable-partition transition to FRI index_version {}; \
                     upgrade to a newer version of Lance",
                    entry.index_version
                )));
            }
            (
                load_raw_frag_reuse_content(dataset, entry).await?,
                entry.fragment_bitmap.clone().unwrap_or_default(),
                Some(entry.dataset_version),
            )
        }
    };
    // A transition is recorded once. The rewrite's intent was diffed against
    // the entry at its read version; the current entry may have moved on
    // (another writer's transitions, a trim), but it can only hold one of
    // ours if the same sources were consumed twice, which the resolver
    // refuses before this point. Check anyway rather than record it twice.
    let recorded = decode_records(&content)?;
    for transition in transitions {
        if recorded
            .transitions
            .iter()
            .any(|existing| transition_identity(existing) == transition_identity(transition))
        {
            return Err(Error::invalid_input(format!(
                "the fragment reuse history already records a transition from fragments {:?}; \
                 the rewrite's entry cannot append it again",
                transition_identity(transition).0
            )));
        }
    }
    for transition in transitions {
        // Another `InlineContent.transitions` (field 2) element; repeated
        // protobuf fields concatenate, so appending preserves the existing
        // wire form untouched.
        content.extend_from_slice(&encode_length_delimited_field(
            2,
            &transition.encode_to_vec(),
        ));
    }

    // Commit-side validation of the assembled entry: lineage order, digest
    // conservation, single content field, mapping presence, unknown-mapping
    // detection. Runs on the inline form before any spill.
    let assembled = prost_types::Any {
        type_url: "/lance.table.FragmentReuseIndexDetails".into(),
        value: encode_length_delimited_field(1, &content),
    };
    let ledger = lance_table::system_index::frag_reuse::ledger::FragReuseLedger::decode(
        1,
        &assembled,
        |_| async {
            Err(Error::invalid_input(
                "the assembled FRI content is inline; no external read is possible",
            ))
        },
    )
    .await?;
    // The decode above uses READER semantics, which deliberately skip
    // transitions with unknown mappings instead of failing; a writer must
    // not maintain a history it cannot fully interpret (spec: "Writers must
    // reject operations that require interpreting or maintaining unsupported
    // mappings").
    if ledger.has_unsupported_transitions() {
        return Err(Error::not_supported(
            "the fragment reuse history contains mappings this writer cannot maintain;              upgrade to a newer version of Lance before rewriting this table",
        ));
    }
    // Every stable-partition transition must own its row map: a reused
    // map_id would let maintenance of one transition delete or overwrite the
    // map file another live transition still references (object-store
    // writes are not create-if-absent, so nothing else catches the clash).
    let mut seen_map_ids = HashSet::new();
    for transition in ledger.transitions() {
        if let lance_table::system_index::frag_reuse::ledger::Mapping::StablePartition(partition) =
            transition.mapping()
            && !seen_map_ids.insert(partition.map_id.clone())
        {
            return Err(Error::invalid_input(format!(
                "stable-partition row-map id {} is referenced by more than one transition;                  each transition must own its row map",
                partition.map_id
            )));
        }
    }

    // Provenance: the previous coverage plus every fragment this rewrite's
    // transitions touch, retired sources deliberately included.
    let mut fragment_bitmap = base_bitmap;
    for transition in transitions {
        for digest in transition
            .sources
            .iter()
            .chain(transition.destinations.iter())
        {
            // In-range by construction: the ledger decode above already
            // validated every digest id against the row-address fragment
            // bound (the one authoritative enforcement point).
            fragment_bitmap.insert(digest.id as u32);
        }
    }

    let entry = build_tagged_frag_reuse_entry(dataset, content, fragment_bitmap).await?;
    Ok((entry, base_entry_version))
}
