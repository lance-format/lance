// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::Dataset;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexInternalExt;
use crate::index::frag_reuse::{
    build_frag_reuse_index_metadata, build_tagged_frag_reuse_entry,
    decode_frag_reuse_ledger_from_content, load_frag_reuse_index_details,
    load_raw_frag_reuse_content,
};
use lance_core::{Error, Result};
use lance_index::frag_reuse::{
    CompactFragReuseIndex, FRAG_REUSE_INDEX_NAME, FragReuseIndexDetails, FragReuseVersion,
};
use lance_index::is_system_index;
use lance_index::metrics::NoOpMetricsCollector;
use lance_table::format::IndexMetadata;
use lance_table::format::pb::fragment_reuse_index_details as pb_fri;
use lance_table::io::manifest::read_manifest_indexes;
use log::warn;
use prost::Message;
use roaring::RoaringBitmap;

impl Dataset {
    /// Opens the fragment reuse index (FRI) recorded in the dataset version this
    /// handle has loaded, or `None` if that version has none. The index is served
    /// from the session index cache, so repeated calls do not re-read it.
    ///
    /// The FRI records how physical row addresses moved in compactions run with
    /// [`CompactionOptions::defer_index_remap`]: one reuse version per compaction,
    /// applied oldest to newest. [`CompactFragReuseIndex::row_addr_remap`] gives
    /// the raw per-address result:
    ///
    /// * `None`: not mapped by any retained version. Helpers such as
    ///   [`CompactFragReuseIndex::remap_row_id`] return these unchanged.
    /// * `Some(None)`: deleted by a recorded compaction.
    /// * `Some(Some(addr))`: the last address reached through the retained
    ///   mappings, not a validated current location.
    ///
    /// # Limitations
    ///
    /// * Covers only the loaded version; check out a newer version to observe
    ///   later compactions.
    /// * Versions are trimmed by [`cleanup_frag_reuse_index`] once indices catch
    ///   up, so an unmapped address may still have moved.
    /// * Mapped destinations are not checked against the manifest and can be
    ///   stale, for example after every row of the destination fragment is deleted.
    /// * Not every compaction records an FRI: it requires `defer_index_remap`,
    ///   fresh index-free tables do not receive one automatically, and datasets
    ///   with stable row ids reject the option.
    /// * Says nothing about deletion files, source-value changes, or whether an
    ///   address belongs to this table or branch.
    ///
    /// Prefer the remap methods over [`CompactFragReuseIndex::details`], which
    /// mirrors the persisted format and is not a long-term client contract.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example(dataset: &lance::Dataset) -> lance::Result<()> {
    /// let old_row_addr: u64 = 42;
    /// if let Some(frag_reuse_index) = dataset.frag_reuse_index().await? {
    ///     match frag_reuse_index.row_addr_remap().get(old_row_addr) {
    ///         None => println!("no recorded movement for {old_row_addr}"),
    ///         Some(None) => println!("row {old_row_addr} was deleted by compaction"),
    ///         Some(Some(new_row_addr)) => println!("row moved to {new_row_addr}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`CompactionOptions::defer_index_remap`]: crate::dataset::optimize::CompactionOptions::defer_index_remap
    pub async fn frag_reuse_index(&self) -> Result<Option<Arc<CompactFragReuseIndex>>> {
        self.open_frag_reuse_index(&NoOpMetricsCollector).await
    }
}

/// Cleanup a fragment reuse index based on the current condition of the indices.
/// If all the indices currently available are already caught up to as a specific reuse version,
/// all older reuse versions (inclusive) can be cleaned up.
///
/// An index is considered caught up against a specific reuse version if either:
/// 1. its coverage is disjoint from the fragments the reuse chain touches, so it
///    holds nothing the FRI would remap (the common multi-index case: a
///    compaction rewrote a sibling index's fragments, not this one); or
/// 2. it is at or past the reuse version's dataset version and no old fragment
///    in the version is still in its bitmap. A missing bitmap counts as caught
///    up, else the version could never be cleaned up.
///
/// Note that there could be a race condition that an index is being added during the cleanup,
/// This will make that specific index not efficient until the next reindex,
/// but it will not cause any correctness problem.
///
/// Typically run after [`compact_files`] with deferred remap and per-index
/// [`remap_column_index`] have caught the indexes up.
///
/// # Example
///
/// ```no_run
/// # use lance::dataset::index::frag_reuse::cleanup_frag_reuse_index;
/// # async fn example(dataset: &mut lance::Dataset) -> lance::Result<()> {
/// // Trim the fragment-reuse index to the versions still needed by some index.
/// cleanup_frag_reuse_index(dataset).await?;
/// # Ok(())
/// # }
/// ```
///
/// [`compact_files`]: crate::dataset::optimize::compact_files
/// [`remap_column_index`]: crate::dataset::optimize::remapping::remap_column_index
pub async fn cleanup_frag_reuse_index(dataset: &mut Dataset) -> lance_core::Result<()> {
    // check against index metadata before auto-remap
    let indices = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await?;
    let Some(frag_reuse_index_meta) = indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
    else {
        return Ok(());
    };

    // Hard fork by index_version: tagged histories trim per transition in
    // fresh code below, while the v0 path stays exactly as it was.
    if frag_reuse_index_meta.index_version != 0 {
        return cleanup_tagged_frag_reuse_index(dataset).await;
    }

    let frag_reuse_details = load_frag_reuse_index_details(dataset, frag_reuse_index_meta).await?;

    let chain_frag_bitmap = reuse_chain_frag_bitmap(&frag_reuse_details.versions);

    let mut retained_versions = Vec::new();
    let mut fragment_bitmaps = RoaringBitmap::new();
    for version in frag_reuse_details.versions.iter() {
        let check_results = indices
            .iter()
            .map(|idx| is_index_remap_caught_up(version, idx, &chain_frag_bitmap))
            .collect::<Vec<_>>();

        if check_results
            .iter()
            .any(|r| matches!(r, Err(Error::InvalidInput { .. })))
        {
            // If the check fails, the reuse version is likely corrupted, do not retain it.
            continue;
        }

        if !check_results.into_iter().all(|r| r.unwrap()) {
            fragment_bitmaps.extend(version.new_frag_bitmap());
            retained_versions.push(version.clone());
        }
    }

    // Return early if there is nothing to cleanup
    if retained_versions.len() == frag_reuse_details.versions.len() {
        return Ok(());
    }

    let frag_reuse_index_details = FragReuseIndexDetails {
        versions: retained_versions,
    };

    let new_index_meta = build_frag_reuse_index_metadata(
        dataset,
        Some(frag_reuse_index_meta),
        frag_reuse_index_details,
        fragment_bitmaps,
    )
    .await?;

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![new_index_meta],
            removed_indices: vec![frag_reuse_index_meta.clone()],
        },
        None,
    );

    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(())
}

/// Every fragment the reuse chain touches (old + new) across all versions. An
/// index disjoint from this set holds no row address the FRI remaps, so trimming
/// can never strand it (fragment ids are never reused).
fn reuse_chain_frag_bitmap(versions: &[FragReuseVersion]) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    for version in versions {
        bitmap.extend(version.old_frag_ids().iter().map(|&id| id as u32));
        bitmap.extend(version.new_frag_ids().iter().map(|&id| id as u32));
    }
    bitmap
}

fn is_index_remap_caught_up(
    frag_reuse_version: &FragReuseVersion,
    index_meta: &IndexMetadata,
    chain_frag_bitmap: &RoaringBitmap,
) -> lance_core::Result<bool> {
    if is_system_index(index_meta) {
        return Ok(true);
    }

    // Disjoint coverage => caught up regardless of dataset_version, bypassing the
    // stale-version gate below (see fn docs). The chain includes NEW fragments
    // deliberately: a deferred-remap commit advances a covering index's bitmap
    // onto them before its data is remapped, so an old-frag-only check would
    // clear a still-stale index and trim a version it needs.
    if let Some(index_frag_bitmap) = &index_meta.fragment_bitmap
        && index_frag_bitmap.is_disjoint(chain_frag_bitmap)
    {
        return Ok(true);
    }

    if index_meta.dataset_version < frag_reuse_version.dataset_version {
        return Ok(false);
    }

    match index_meta.fragment_bitmap.clone() {
        Some(index_frag_bitmap) => {
            for group in frag_reuse_version.groups.iter() {
                let mut old_frag_in_index = 0;
                for old_frag in group.old_frags.iter() {
                    if index_frag_bitmap.contains(old_frag.id as u32) {
                        old_frag_in_index += 1;
                    }
                }

                if old_frag_in_index > 0 {
                    if old_frag_in_index != group.old_frags.len() {
                        // This should never happen because we always commit a full rewrite group
                        // and we always reindex either the entire group or nothing.
                        // We use invalid input to be consistent with
                        // dataset::transaction::recalculate_fragment_bitmap
                        return Err(Error::invalid_input(format!(
                            "The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {:?}",
                            group.old_frags
                        )));
                    }
                    return Ok(false);
                }
            }
            Ok(true)
        }
        None => {
            warn!(
                "Index {} ({}) missing fragment bitmap, cannot determine if it is caught up with the fragment reuse version, consider retraining the index",
                index_meta.name, index_meta.uuid
            );
            Ok(true)
        }
    }
}

/// Conflict-source marker for a trim whose rebase derived nothing left to
/// trim: a concurrent commit already satisfied it, so instead of writing an
/// empty no-op manifest version the commit attempt aborts with a retryable
/// conflict carrying this message, and [`cleanup_frag_reuse_index`] treats
/// exactly that conflict as success.
pub(crate) const TAGGED_TRIM_REBASED_TO_NOOP: &str =
    "the tagged trim rebased to nothing to trim; a concurrent commit already satisfied it";

/// The outcome of deriving a tagged trim against the current manifest.
#[derive(Debug)]
pub(crate) enum TaggedTrimOutcome {
    /// Every record is still needed, or there is no tagged entry to trim.
    NothingToTrim,
    /// Replace the current entry with the filtered history.
    Replace {
        new_entry: IndexMetadata,
        current_entry: IndexMetadata,
    },
    /// Everything trimmed away: delete the entry outright. The manifest
    /// feature flags stay set (they are sticky).
    Delete { current_entry: IndexMetadata },
}

/// Segments of a logical index whose whole contribution is already served
/// directly by other retained segments of the same index.
///
/// A segment X is superseded when every fragment of its VALID COVERAGE --
/// (stored bitmap ∩ live fragments) ∪ (the destinations X reaches through the
/// transition chain, exactly as the reader derives them via `load_indices`) --
/// is directly covered by kept segments of the same logical index. Both
/// halves matter: a segment still directly covering a live fragment nobody
/// else covers is NOT superseded even if all its translated destinations are
/// taken over.
///
/// Safe by direct-coverage-wins: the reader already masks X's contribution
/// for every fragment another segment covers directly, so removing a
/// qualifying X cannot change any query result.
///
/// Deterministic fixed point for mutual redundancy: segments are confirmed
/// newest-first (dataset_version, then uuid), and a segment is prunable only
/// against the direct coverage of segments already confirmed KEPT -- so of
/// two segments fully covering each other, the newest survives, and at least
/// one segment of every logical index always remains.
pub(crate) async fn derive_superseded_segments(
    dataset: &Dataset,
) -> lance_core::Result<Vec<IndexMetadata>> {
    let stored = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await?;
    // The reader's own coverage derivation (translated through the ledger,
    // direct-coverage-wins applied); not reimplemented here.
    use crate::index::DatasetIndexExt;
    let derived = dataset.load_indices().await?;
    let derived_by_uuid: HashMap<uuid::Uuid, &RoaringBitmap> = derived
        .iter()
        .filter_map(|idx| {
            idx.fragment_bitmap
                .as_ref()
                .map(|bitmap| (idx.uuid, bitmap))
        })
        .collect();
    let live = dataset.fragment_bitmap.as_ref();

    let mut groups: HashMap<&str, Vec<&IndexMetadata>> = HashMap::new();
    for index in stored.iter() {
        if is_system_index(index) {
            continue;
        }
        groups.entry(index.name.as_str()).or_default().push(index);
    }

    let mut removals = Vec::new();
    for segments in groups.into_values() {
        let mut segments = segments;
        // Newest first; uuid as a stable tie-break.
        segments.sort_by(|a, b| {
            b.dataset_version
                .cmp(&a.dataset_version)
                .then_with(|| b.uuid.cmp(&a.uuid))
        });
        let mut kept_direct = RoaringBitmap::new();
        let mut kept_any = false;
        for segment in segments {
            // A segment this build cannot serve (unknown type or newer index
            // version) is opaque: its translated coverage is invisible, so it
            // is never pruned, and it must not mask anyone -- the reader will
            // not answer queries from it, so its bitmap covers nothing.
            if !crate::index::index_is_usable(segment) {
                kept_any = true;
                continue;
            }
            let Some(stored_bitmap) = segment.fragment_bitmap.as_ref() else {
                // Unknown coverage cannot be reasoned about; always keep.
                kept_any = true;
                continue;
            };
            let direct_live = stored_bitmap & live;
            let mut valid = direct_live;
            if let Some(translated) = derived_by_uuid.get(&segment.uuid) {
                valid |= *translated & live;
            }
            if kept_any && valid.is_subset(&kept_direct) {
                removals.push(segment.clone());
            } else {
                kept_any = true;
                kept_direct |= stored_bitmap & live;
            }
        }
    }
    Ok(removals)
}

/// Remove superseded segments, re-deriving on every attempt: a conflicting
/// concurrent commit (e.g. index maintenance under the same name) refreshes
/// the dataset and the removal set is recomputed against the new state.
async fn prune_superseded_segments(dataset: &mut Dataset) -> lance_core::Result<()> {
    const MAX_ATTEMPTS: usize = 5;
    let mut last_conflict = None;
    for _ in 0..MAX_ATTEMPTS {
        let removals = derive_superseded_segments(dataset).await?;
        if removals.is_empty() {
            return Ok(());
        }
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![],
                removed_indices: removals,
            },
            None,
        );
        match dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
        {
            Ok(()) => return Ok(()),
            Err(error @ Error::RetryableCommitConflict { .. }) => {
                dataset.checkout_latest().await?;
                last_conflict = Some(error);
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_conflict.expect("loop only exits with a recorded conflict"))
}

/// Trim a tagged FRI entry, deriving what to keep from the CURRENT manifest.
///
/// The derivation runs again inside the commit path on every retry (see the
/// conflict resolver's `finish_create_index`), so a concurrent append lands in
/// the re-derived entry instead of being spliced away.
async fn cleanup_tagged_frag_reuse_index(dataset: &mut Dataset) -> lance_core::Result<()> {
    // Superseded segments go first, so the trim below derives against the
    // pruned world and can release transitions that only those segments
    // needed, in the same maintenance invocation.
    prune_superseded_segments(dataset).await?;

    let operation = match derive_tagged_trim(dataset).await? {
        TaggedTrimOutcome::NothingToTrim => return Ok(()),
        TaggedTrimOutcome::Replace {
            new_entry,
            current_entry,
        } => Operation::CreateIndex {
            new_indices: vec![new_entry],
            removed_indices: vec![current_entry],
        },
        TaggedTrimOutcome::Delete { current_entry } => Operation::CreateIndex {
            new_indices: vec![],
            removed_indices: vec![current_entry],
        },
    };

    let transaction = Transaction::new(dataset.manifest.version, operation, None);
    match dataset
        .apply_commit(
            transaction,
            &crate::dataset::ManifestWriteConfig::default().with_tagged_frag_reuse_trim(),
            &Default::default(),
        )
        .await
    {
        Ok(()) => Ok(()),
        Err(error @ Error::RetryableCommitConflict { .. })
            if error.to_string().contains(TAGGED_TRIM_REBASED_TO_NOOP) =>
        {
            // The rebase found nothing left to trim: a concurrent commit
            // (e.g. an index catching up mid-trim) already satisfied this
            // maintenance run, so succeed without writing a version.
            dataset.checkout_latest().await?;
            Ok(())
        }
        Err(error) => Err(error),
    }
}

/// Whether a `CreateIndex` payload has the tagged-trim shape: exactly the
/// tagged FRI entry replaced, or removed outright. A v0 trim (v0 entry
/// replacing a v0 entry) deliberately does not match.
pub(crate) fn is_tagged_trim_operation(
    new_indices: &[IndexMetadata],
    removed_indices: &[IndexMetadata],
) -> bool {
    use lance_table::system_index::frag_reuse::metadata::is_tagged;
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

/// One splice unit of the entry's content: a legacy version (field 1) or a
/// transition (field 2), kept with its exact wire form so retained records
/// are carried forward verbatim.
enum TrimRecord {
    LegacyVersion(FragReuseVersion),
    Transition(pb_fri::Transition),
}

struct TrimElement {
    raw: Vec<u8>,
    record: TrimRecord,
}

fn split_trim_elements(content: &[u8]) -> Result<Vec<TrimElement>> {
    use bytes::Buf;
    use prost::encoding::{WireType, decode_key, decode_varint};

    let corrupt = |message: String| Error::corrupt_file_named("FRI details", message);
    let total = content.len();
    let mut buf = bytes::Bytes::copy_from_slice(content);
    let mut elements = Vec::new();
    while buf.has_remaining() {
        let start = total - buf.remaining();
        let (tag, wire_type) = decode_key(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        // Fields 1 and 2 are the only known records and both are messages. An
        // unknown record may participate in lineage in ways this client cannot
        // reason about, so refuse to trim rather than silently dropping or
        // blindly retaining it.
        if wire_type != WireType::LengthDelimited || !matches!(tag, 1 | 2) {
            return Err(Error::not_supported(format!(
                "the tagged FRI history carries an unknown record (field {tag}); \
                 upgrade to a newer version of Lance before trimming"
            )));
        }
        let length = decode_varint(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        if length > buf.remaining() as u64 {
            return Err(corrupt(format!(
                "field {tag} length {length} exceeds remaining {} bytes",
                buf.remaining()
            )));
        }
        let payload = buf.split_to(length as usize);
        let end = total - buf.remaining();
        let record = match tag {
            1 => TrimRecord::LegacyVersion(
                pb_fri::Version::decode(payload)
                    .map_err(|e| corrupt(e.to_string()))?
                    .try_into()?,
            ),
            _ => TrimRecord::Transition(
                pb_fri::Transition::decode(payload).map_err(|e| corrupt(e.to_string()))?,
            ),
        };
        elements.push(TrimElement {
            raw: content[start..end].to_vec(),
            record,
        });
    }
    Ok(elements)
}

/// One logical index's coverage as the trim sees it: provenance is the union
/// of every stored segment bitmap (any segment, usable by this build or not,
/// pins the records its addresses need), while direct coverage counts only
/// segments this build can actually serve (`index_is_usable`) -- an unusable
/// segment answers no query, so its bitmap must not mask anyone's need for a
/// translation.
#[derive(Default)]
struct IndexGroupCoverage {
    provenance: RoaringBitmap,
    direct: RoaringBitmap,
}

/// Whether some index still needs this transition to translate its rows.
///
/// RETAIN iff there is a logical index I (its stored segments grouped by
/// name, system indices exempt) such that:
/// * some segment's stored provenance bitmap intersects the transition's
///   sources (I holds addresses the transition moves), AND
/// * some destination is not directly covered by any USABLE segment of I
///   (I has not fully re-derived onto the transition's output in a form this
///   build can serve).
///
/// A segment without a stored bitmap imposes no constraint, mirroring the v0
/// leniency for missing bitmaps.
fn is_transition_needed(
    transition: &pb_fri::Transition,
    index_groups: &HashMap<&str, IndexGroupCoverage>,
) -> bool {
    index_groups.values().any(|group| {
        let sources_hit = transition
            .sources
            .iter()
            .any(|source| group.provenance.contains(source.id as u32));
        sources_hit
            && transition
                .destinations
                .iter()
                .any(|destination| !group.direct.contains(destination.id as u32))
    })
}

/// Which elements of a tagged history are still needed, from the entry's
/// digests and the stored index metadata alone (zero payload IO).
///
/// Legacy versions keep the exact v0 caught-up predicate; transitions use
/// [`is_transition_needed`]; then transitive chain retention closes the set
/// forward: a retained record's translation path walks through every
/// downstream consumer of its destinations, so those records must survive
/// too. Legacy versions retain at whole-version granularity (their groups
/// stand or fall together, as in v0).
fn compute_tagged_retention(elements: &[TrimElement], indices: &[IndexMetadata]) -> Vec<bool> {
    // The v0 predicate's disjointness rule is scoped to the legacy chain, as
    // in v0: an index touching only tagged-transition fragments holds nothing
    // a legacy version remaps.
    let legacy_versions: Vec<FragReuseVersion> = elements
        .iter()
        .filter_map(|element| match &element.record {
            TrimRecord::LegacyVersion(version) => Some(version.clone()),
            TrimRecord::Transition(_) => None,
        })
        .collect();
    let chain_frag_bitmap = reuse_chain_frag_bitmap(&legacy_versions);

    // Stored segments grouped by logical index name. Every segment's bitmap
    // counts as provenance (pinning); only usable segments' bitmaps count as
    // direct coverage (masking).
    let mut index_groups: HashMap<&str, IndexGroupCoverage> = HashMap::new();
    for index in indices.iter() {
        if is_system_index(index) {
            continue;
        }
        let group = index_groups.entry(index.name.as_str()).or_default();
        match &index.fragment_bitmap {
            Some(bitmap) => {
                group.provenance |= bitmap;
                if crate::index::index_is_usable(index) {
                    group.direct |= bitmap;
                }
            }
            None => warn!(
                "Index {} ({}) missing fragment bitmap, it cannot pin tagged fragment reuse records, consider retraining the index",
                index.name, index.uuid
            ),
        }
    }

    let mut retained = vec![false; elements.len()];
    for (position, element) in elements.iter().enumerate() {
        match &element.record {
            TrimRecord::LegacyVersion(version) => {
                let check_results = indices
                    .iter()
                    .map(|idx| is_index_remap_caught_up(version, idx, &chain_frag_bitmap))
                    .collect::<Vec<_>>();
                if check_results
                    .iter()
                    .any(|r| matches!(r, Err(Error::InvalidInput { .. })))
                {
                    // If the check fails, the reuse version is likely corrupted,
                    // do not retain it (v0 behavior).
                    continue;
                }
                if !check_results.into_iter().all(|r| r.unwrap()) {
                    retained[position] = true;
                }
            }
            TrimRecord::Transition(transition) => {
                retained[position] = is_transition_needed(transition, &index_groups);
            }
        }
    }

    // Transitive chain retention (forward closure over producer->consumer
    // edges).
    struct Node {
        element: usize,
        sources: Vec<u64>,
        destinations: Vec<u64>,
    }
    let mut nodes = Vec::new();
    for (position, element) in elements.iter().enumerate() {
        match &element.record {
            TrimRecord::LegacyVersion(version) => {
                for group in version.groups.iter() {
                    nodes.push(Node {
                        element: position,
                        sources: group.old_frags.iter().map(|f| f.id).collect(),
                        destinations: group.new_frags.iter().map(|f| f.id).collect(),
                    });
                }
            }
            TrimRecord::Transition(transition) => nodes.push(Node {
                element: position,
                sources: transition.sources.iter().map(|d| d.id).collect(),
                destinations: transition.destinations.iter().map(|d| d.id).collect(),
            }),
        }
    }
    let mut consumer_of: HashMap<u64, usize> = HashMap::new();
    let mut element_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node_index, node) in nodes.iter().enumerate() {
        for source in &node.sources {
            consumer_of.insert(*source, node_index);
        }
        element_nodes
            .entry(node.element)
            .or_default()
            .push(node_index);
    }
    let mut node_retained = vec![false; nodes.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (node_index, node) in nodes.iter().enumerate() {
        if retained[node.element] {
            node_retained[node_index] = true;
            queue.push_back(node_index);
        }
    }
    while let Some(node_index) = queue.pop_front() {
        for destination in nodes[node_index].destinations.clone() {
            let Some(&consumer) = consumer_of.get(&destination) else {
                continue;
            };
            if node_retained[consumer] {
                continue;
            }
            node_retained[consumer] = true;
            queue.push_back(consumer);
            let element = nodes[consumer].element;
            if !retained[element] {
                retained[element] = true;
                for &sibling in element_nodes[&element].iter() {
                    if !node_retained[sibling] {
                        node_retained[sibling] = true;
                        queue.push_back(sibling);
                    }
                }
            }
        }
    }

    retained
}

/// Derive the trimmed tagged entry from the CURRENT manifest state.
///
/// Zero payload IO: retention is computed from the entry's digests and the
/// stored index metadata alone. Row-map files released by trimming a
/// stable-partition transition are collected separately by dataset cleanup.
pub(crate) async fn derive_tagged_trim(dataset: &Dataset) -> lance_core::Result<TaggedTrimOutcome> {
    let indices = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
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
    let retained = compute_tagged_retention(&elements, &indices);

    if retained.iter().all(|kept| *kept) {
        return Ok(TaggedTrimOutcome::NothingToTrim);
    }
    if retained.iter().all(|kept| !*kept) {
        return Ok(TaggedTrimOutcome::Delete {
            current_entry: entry,
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
        new_entry,
        current_entry: entry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::optimize::{CompactionOptions, compact_files, remapping};
    use crate::index::DatasetIndexExt;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use all_asserts::{assert_false, assert_true};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type};
    use lance_core::ROW_ADDR;
    use lance_core::utils::address::RowAddress;
    use lance_datagen::Dimension;
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;
    use std::collections::HashMap;

    fn frag_digest(id: u64) -> lance_index::frag_reuse::FragDigest {
        lance_index::frag_reuse::FragDigest {
            id,
            physical_rows: 100,
            num_deleted_rows: 0,
        }
    }

    fn reuse_version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
        FragReuseVersion {
            dataset_version,
            groups: vec![lance_index::frag_reuse::FragReuseGroup {
                changed_row_addrs: Vec::new(),
                old_frags: old.iter().copied().map(frag_digest).collect(),
                new_frags: new.iter().copied().map(frag_digest).collect(),
            }],
        }
    }

    fn index_covering(dataset_version: u64, covered: &[u32]) -> IndexMetadata {
        IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            fields: vec![0],
            covering_fields: vec![],
            name: "test_idx".into(),
            dataset_version,
            fragment_bitmap: Some(RoaringBitmap::from_iter(covered.iter().copied())),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    /// The catch-up determination must not pin the FRI on an index that is
    /// simply unrelated to the compaction, while still retaining versions that a
    /// covering-but-not-yet-remapped index needs.
    #[test]
    fn test_caught_up_uses_fragment_coverage_not_only_version() {
        // A reuse version at dataset_version 10 rewrote fragments [4, 5] -> [6].
        let version = reuse_version(10, &[4, 5], &[6]);
        let chain = reuse_chain_frag_bitmap(std::slice::from_ref(&version));

        // Non-covering, stale version: touches none of the rewritten frags, so
        // caught up despite version 5 < 10 (the case the old gate got wrong).
        assert_true!(
            is_index_remap_caught_up(&version, &index_covering(5, &[1, 2, 3]), &chain).unwrap()
        );

        // Still holds an old fragment: not caught up.
        assert_false!(
            is_index_remap_caught_up(&version, &index_covering(5, &[1, 4, 5]), &chain).unwrap()
        );

        // Bitmap advanced onto the new fragment but data not yet remapped: not
        // caught up (why the chain must include new frags).
        assert_false!(
            is_index_remap_caught_up(&version, &index_covering(5, &[1, 6]), &chain).unwrap()
        );

        // Once remapped (version advanced): caught up.
        assert_true!(
            is_index_remap_caught_up(&version, &index_covering(11, &[1, 6]), &chain).unwrap()
        );
    }

    /// The chain spans every reuse version, not just the one being checked: a
    /// stale index touching only a *later* version's fragment must still fall to
    /// the version gate (a per-version chain would wrongly clear it).
    #[test]
    fn test_caught_up_uses_whole_reuse_chain() {
        let v1 = reuse_version(10, &[4, 5], &[6]); // 4,5 -> 6
        let v2 = reuse_version(11, &[6], &[7]); // 6 -> 7
        let chain = reuse_chain_frag_bitmap(&[v1.clone(), v2]);

        // Stale index (version 5) covering only v2's new fragment [7]: not
        // disjoint from the chain, so not caught up on v1.
        assert_false!(is_index_remap_caught_up(&v1, &index_covering(5, &[1, 7]), &chain).unwrap());
    }

    /// Whole-fragment removal (every row deleted, no replacement): an index
    /// emptied by the deletion has an empty bitmap and must count as caught up --
    /// it holds only dead rows -- else its stale version pins the removed-fragment
    /// version forever (remap hits the drop-everything path, never advancing it).
    #[test]
    fn test_caught_up_handles_fragment_removal() {
        // Reuse version 20 removed fragment [7] outright (no replacement).
        let version = reuse_version(20, &[7], &[]);
        let chain = reuse_chain_frag_bitmap(std::slice::from_ref(&version));

        // Index emptied by the deletion (empty bitmap): caught up.
        assert_true!(is_index_remap_caught_up(&version, &index_covering(5, &[]), &chain).unwrap());

        // Bitmap still lists the removed fragment (not yet updated): retained.
        assert_false!(
            is_index_remap_caught_up(&version, &index_covering(5, &[7]), &chain).unwrap()
        );
    }

    #[tokio::test]
    async fn test_cleanup_frag_reuse_index() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Create an index to be remapped
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Compact and check index not caught up
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);
        let indices = dataset.load_indices().await.unwrap();
        let scalar_index = indices.iter().find(|idx| idx.name == "scalar").unwrap();
        // Should not be considered caught up because index was created at an old dataset version
        assert_false!(
            is_index_remap_caught_up(
                &frag_reuse_details.versions[0],
                scalar_index,
                &reuse_chain_frag_bitmap(&frag_reuse_details.versions),
            )
            .unwrap()
        );

        // Remap and check index is caught up
        remapping::remap_column_index(&mut dataset, &["i"], index_name.clone())
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let scalar_index = indices.iter().find(|idx| idx.name == "scalar").unwrap();
        assert_true!(
            is_index_remap_caught_up(
                &frag_reuse_details.versions[0],
                scalar_index,
                &reuse_chain_frag_bitmap(&frag_reuse_details.versions),
            )
            .unwrap()
        );

        // Cleanup frag reuse index and check there is no reuse version
        let mut dataset_clone = dataset.clone();
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 0);

        // Try doing a concurrent cleanup should fail with conflict
        assert!(matches!(
            cleanup_frag_reuse_index(&mut dataset_clone).await,
            Err(Error::RetryableCommitConflict { .. })
        ));
    }

    /// With more than one index on the table, remapping every index must catch
    /// all of them up so the reuse index can be trimmed.
    ///
    /// Regression: `remap_column_index` used to decide whether to remap an
    /// index's data from the presence of the old fragments in its fragment
    /// bitmap. But `load_indices` coverage-remaps the bitmap onto the new
    /// fragments in memory, and remapping the *first* index commits a manifest
    /// that persists that cleaned bitmap for the others — so remapping the
    /// remaining indexes became a silent no-op (their data was never remapped
    /// and their `dataset_version` never advanced), and the reuse index could
    /// never be trimmed.
    #[tokio::test]
    async fn test_cleanup_frag_reuse_index_multiple_indices() {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .col("j", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        for col in ["i", "j"] {
            dataset
                .create_index(
                    &[col],
                    IndexType::Scalar,
                    Some(format!("{col}_idx")),
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        let frag_reuse_index_meta = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .expect("Fragment reuse index must be available");
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        for col in ["i", "j"] {
            remapping::remap_column_index(&mut dataset, &[col], Some(format!("{col}_idx")))
                .await
                .unwrap();
        }

        // Every index must now be caught up (data remapped, version advanced).
        let indices = dataset.load_indices().await.unwrap();
        for col in ["i", "j"] {
            let index = indices
                .iter()
                .find(|idx| idx.name == format!("{col}_idx"))
                .unwrap();
            assert!(
                is_index_remap_caught_up(
                    &frag_reuse_details.versions[0],
                    index,
                    &reuse_chain_frag_bitmap(&frag_reuse_details.versions),
                )
                .unwrap(),
                "index {col}_idx was not caught up after remap"
            );
        }

        // ... so the reuse index trims down to zero versions.
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();
        let frag_reuse_index_meta = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .expect("Fragment reuse index must be available");
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 0);

        // Data correctness, not just version bookkeeping: with the reuse index
        // trimmed there is no auto-remap safety net, so each index must resolve
        // to LIVE rows. An index whose data was not actually remapped (e.g. one
        // whose bitmap was coverage-remapped by a sibling's commit before its
        // own data remap) points at compacted-away fragments and errors on take.
        use futures::TryStreamExt;
        for col in ["i", "j"] {
            let rows: usize = dataset
                .scan()
                .filter(&format!("{col} >= 2000 AND {col} < 3000"))
                .unwrap()
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .iter()
                .map(|b| b.num_rows())
                .sum();
            assert_eq!(
                rows, 1000,
                "index {col}_idx must resolve to live rows after remap+trim"
            );
        }
    }

    /// When the reuse index has accumulated several versions, a single remap
    /// must compose them and rebuild + commit the index exactly ONCE, not once
    /// per version.
    #[tokio::test]
    async fn test_remap_index_batches_multiple_reuse_versions() {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(8), FragmentRowCount::from(1000))
            .await
            .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("i_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Accumulate multiple reuse versions: each round deletes a prefix, which
        // shrinks fragments below target and forces another deferred compaction.
        let options = CompactionOptions {
            target_rows_per_fragment: 4_000,
            defer_index_remap: true,
            ..Default::default()
        };
        for round in 0..4 {
            dataset
                .delete(&format!("i < {}", 1_000 * (round + 1)))
                .await
                .unwrap();
            compact_files(&mut dataset, options.clone(), None)
                .await
                .unwrap();
        }

        let frag_reuse_index_meta = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .expect("Fragment reuse index must be available");
        let num_versions = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap()
            .versions
            .len();
        assert!(
            num_versions >= 2,
            "test needs multiple reuse versions to exercise batching, got {num_versions}"
        );

        // A single remap must commit exactly once, regardless of version count.
        let version_before = dataset.manifest.version;
        remapping::remap_column_index(&mut dataset, &["i"], Some("i_idx".into()))
            .await
            .unwrap();
        let commits = dataset.manifest.version - version_before;
        assert_eq!(
            commits, 1,
            "batched remap must commit once, not once per reuse version ({num_versions})"
        );

        // ... and the reuse index then trims to zero.
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();
        let frag_reuse_index_meta = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .expect("Fragment reuse index must be available");
        assert_eq!(
            load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
                .await
                .unwrap()
                .versions
                .len(),
            0
        );
    }

    mod tagged_trim {
        use super::super::*;
        use crate::dataset::WriteParams;
        use crate::dataset::optimize::{CompactionOptions, compact_files};
        use crate::dataset::write::CommitBuilder;
        use crate::index::DatasetIndexExt;
        use crate::index::frag_reuse::{decode_frag_reuse_ledger, load_raw_frag_reuse_content};
        use crate::index::frag_reuse_reader::tests as reader_tests;
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int32Type;
        use lance_index::IndexType;
        use lance_index::scalar::ScalarIndexParams;
        use lance_table::format::Fragment;
        use lance_table::system_index::frag_reuse::ledger::Mapping;
        use lance_table::transaction::{FragmentReuseRewrite, RewriteGroup};
        use uuid::Uuid;

        fn digest(id: u64) -> pb_fri::FragmentDigest {
            pb_fri::FragmentDigest {
                id,
                physical_rows: 4,
                num_deleted_rows: 0,
            }
        }

        fn transition_element(sources: &[u64], destinations: &[u64]) -> TrimElement {
            TrimElement {
                raw: Vec::new(),
                record: TrimRecord::Transition(pb_fri::Transition {
                    sources: sources.iter().copied().map(digest).collect(),
                    destinations: destinations.iter().copied().map(digest).collect(),
                    mapping: Some(pb_fri::transition::Mapping::StablePartition(
                        pb_fri::StablePartition {
                            map_id: Uuid::new_v4().to_string(),
                            map_size_bytes: 1,
                            base_id: None,
                        },
                    )),
                }),
            }
        }

        fn legacy_element(dataset_version: u64, old: &[u64], new: &[u64]) -> TrimElement {
            TrimElement {
                raw: Vec::new(),
                record: TrimRecord::LegacyVersion(super::reuse_version(dataset_version, old, new)),
            }
        }

        fn named_index(name: &str, dataset_version: u64, covered: &[u32]) -> IndexMetadata {
            let mut index = super::index_covering(dataset_version, covered);
            index.name = name.into();
            index
        }

        /// The per-transition retention rule (B1-B5, B7-B9): retain iff some
        /// logical index still derives coverage from the transition's sources
        /// while not directly covering every destination.
        #[test]
        fn transition_retention_rules() {
            let elements = vec![transition_element(&[1, 2], &[5, 6])];
            let retain =
                |indices: &[IndexMetadata]| compute_tagged_retention(&elements, indices)[0];

            // B1: a disjoint index does not block the trim.
            assert!(!retain(&[named_index("a_idx", 1, &[3, 4])]));
            // B2: still deriving from the sources, destinations uncovered.
            assert!(retain(&[named_index("a_idx", 1, &[1, 2])]));
            // B3: partial drain (one destination still uncovered) retains.
            assert!(retain(&[named_index("a_idx", 1, &[1, 5])]));
            // B4: full direct coverage across the index's segments trims,
            // even while another segment still lists the sources.
            assert!(!retain(&[
                named_index("a_idx", 1, &[1, 2]),
                named_index("a_idx", 2, &[5, 6]),
            ]));
            // B5: segments rebuilt onto the destinations (sources gone) trim.
            assert!(!retain(&[named_index("a_idx", 2, &[5, 6])]));
            // B7: one index caught up, another still hanging off the sources.
            assert!(retain(&[
                named_index("a_idx", 2, &[5, 6]),
                named_index("b_idx", 1, &[1]),
            ]));
            // B8/B9: no indices left, nothing pins the record.
            assert!(!retain(&[]));
            // A missing bitmap imposes no constraint (v0 leniency).
            let mut no_bitmap = named_index("a_idx", 1, &[]);
            no_bitmap.fragment_bitmap = None;
            assert!(!retain(std::slice::from_ref(&no_bitmap)));
            // ... but a sibling segment with a bitmap still pins it.
            assert!(retain(&[no_bitmap, named_index("a_idx", 1, &[1])]));
            // System indices are exempt.
            assert!(!retain(&[named_index(FRAG_REUSE_INDEX_NAME, 1, &[1, 2])]));
        }

        /// B6: chain retention. A segment hanging off the head of a chain
        /// pins every downstream hop its translation path traverses; hops
        /// upstream of where the segment enters are not needed.
        #[test]
        fn chain_retention_closes_forward() {
            let elements = vec![
                transition_element(&[1], &[2]),
                transition_element(&[2], &[3]),
            ];
            // Enters at the head: both hops retained.
            assert_eq!(
                compute_tagged_retention(&elements, &[named_index("a_idx", 1, &[1])]),
                vec![true, true]
            );
            // Enters mid-chain: only the downstream hop retained.
            assert_eq!(
                compute_tagged_retention(&elements, &[named_index("a_idx", 1, &[2])]),
                vec![false, true]
            );
            // Fully retired: both trim together.
            assert_eq!(
                compute_tagged_retention(&elements, &[named_index("a_idx", 2, &[3])]),
                vec![false, false]
            );
        }

        /// Chain retention traverses lifted legacy groups too, and a legacy
        /// version retains at whole-version granularity.
        #[test]
        fn chain_retention_spans_legacy_versions() {
            let elements = vec![
                legacy_element(10, &[1], &[2]),
                transition_element(&[2], &[3]),
            ];
            // A stale index over the legacy sources pins the version (exact
            // v0 predicate) and, transitively, the downstream transition.
            assert_eq!(
                compute_tagged_retention(&elements, &[named_index("a_idx", 5, &[1])]),
                vec![true, true]
            );
            // Caught up past the legacy version and rebuilt onto the final
            // fragments: everything trims.
            assert_eq!(
                compute_tagged_retention(&elements, &[named_index("a_idx", 11, &[3])]),
                vec![false, false]
            );

            // Whole-version granularity: a version with two groups where a
            // transition consumes the second group's output. An index pinning
            // the FIRST group pins the version, whose second group then pins
            // the downstream transition.
            let elements = vec![
                legacy_element(10, &[1, 5], &[2, 6]),
                transition_element(&[6], &[7]),
            ];
            assert_eq!(
                compute_tagged_retention(&elements, &[named_index("a_idx", 5, &[1])]),
                vec![true, true]
            );
        }

        async fn reserve_fragments(dataset: &mut Dataset, num_fragments: u32) {
            dataset
                .apply_commit(
                    Transaction::new(
                        dataset.manifest.version,
                        Operation::ReserveFragments { num_fragments },
                        None,
                    ),
                    &Default::default(),
                    &Default::default(),
                )
                .await
                .unwrap();
        }

        async fn commit_stable_partition(
            dataset: Dataset,
            source_ids: &[u64],
            dest_base_id: u64,
        ) -> Dataset {
            let old_fragments: Vec<Fragment> = source_ids
                .iter()
                .map(|id| {
                    dataset
                        .fragments()
                        .iter()
                        .find(|f| f.id == *id)
                        .unwrap()
                        .clone()
                })
                .collect();
            let (transition, destinations) =
                reader_tests::prepare_partition(&dataset, source_ids, dest_base_id).await;
            let read_version = dataset.manifest.version;
            CommitBuilder::new(Arc::new(dataset))
                .execute(Transaction::new(
                    read_version,
                    Operation::Rewrite {
                        groups: vec![RewriteGroup {
                            old_fragments,
                            new_fragments: destinations,
                        }],
                        rewritten_indices: vec![],
                        frag_reuse_index: None,
                        frag_reuse_rewrite: Some(FragmentReuseRewrite {
                            transitions: vec![transition],
                            base_entry_version: None,
                        }),
                    },
                    None,
                ))
                .await
                .unwrap()
        }

        async fn fri_entry(dataset: &Dataset) -> Option<IndexMetadata> {
            crate::index::load_all_indices(dataset)
                .await
                .unwrap()
                .iter()
                .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                .cloned()
        }

        async fn sorted_values(dataset: &Dataset) -> Vec<i32> {
            let batch = dataset.scan().try_into_batch().await.unwrap();
            let mut values: Vec<i32> = batch["i"]
                .as_primitive::<Int32Type>()
                .iter()
                .map(|value| value.unwrap())
                .collect();
            values.sort_unstable();
            values
        }

        /// The full trim lifecycle on a tagged table: retained while the
        /// index still derives from the sources, deleted outright once the
        /// index is rebuilt over the destinations (C3), with the sticky
        /// feature flags left set and the table still fully usable.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn trim_lifecycle_retains_then_deletes() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 20).await;
            let before = sorted_values(&dataset).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            let entry = fri_entry(&dataset).await.unwrap();
            assert_eq!(entry.index_version, 1);

            // Still deriving: the trim must be a no-op (no commit at all).
            let version_before = dataset.manifest.version;
            cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            assert_eq!(dataset.manifest.version, version_before);
            assert_eq!(fri_entry(&dataset).await.unwrap().uuid, entry.uuid);

            // Drain: rebuild the index over the destinations (the gate must
            // admit user index maintenance on a tagged table).
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("i_idx".into()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();

            // Fully drained: the entry is deleted outright.
            cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            assert!(fri_entry(&dataset).await.is_none());
            let flag = lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
            assert_eq!(dataset.manifest.reader_feature_flags & flag, flag);
            assert_eq!(dataset.manifest.writer_feature_flags & flag, flag);

            // The (still-flagged) table keeps working: reads, filtered reads,
            // and new writes.
            assert_eq!(sorted_values(&dataset).await, before);
            assert_eq!(dataset.count_rows(Some("i >= 4".into())).await.unwrap(), 4);
            let batch = lance_datagen::gen_batch()
                .col("i", lance_datagen::array::step_custom::<Int32Type>(100, 1))
                .into_batch_rows(lance_datagen::RowCount::from(4))
                .unwrap();
            let dataset = crate::dataset::InsertBuilder::new(Arc::new(dataset))
                .with_params(&WriteParams {
                    mode: crate::dataset::WriteMode::Append,
                    ..Default::default()
                })
                .execute(vec![batch])
                .await
                .unwrap();
            assert_eq!(dataset.count_rows(None).await.unwrap(), 12);
        }

        /// A trim committed from a stale read must re-derive at commit time:
        /// a stable-partition rewrite that landed in between keeps its
        /// transition while the drained one is still released.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn trim_rebases_over_concurrent_stable_partition() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 40).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            // Drain the index onto the destinations.
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("i_idx".into()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();

            // A second stable-partition rewrite lands after the trim's read.
            let mut stale = dataset.clone();
            let dataset = commit_stable_partition(dataset, &[10, 11], 20).await;
            let current_entry = fri_entry(&dataset).await.unwrap();

            // The stale trim would have deleted the entry; the rebase must
            // keep the new transition instead.
            cleanup_frag_reuse_index(&mut stale).await.unwrap();
            let trimmed_entry = fri_entry(&stale).await.unwrap();
            assert_ne!(trimmed_entry.uuid, current_entry.uuid);
            let ledger = decode_frag_reuse_ledger(&stale, &trimmed_entry)
                .await
                .unwrap();
            assert_eq!(ledger.transitions().len(), 1);
            let kept = &ledger.transitions()[0];
            assert_eq!(
                kept.sources().iter().map(|d| d.id).collect::<Vec<_>>(),
                vec![10, 11]
            );
            assert!(matches!(kept.mapping(), Mapping::StablePartition(_)));
            // The drained transition's raw bytes are gone; the kept one's
            // content is a verbatim suffix of the previous entry.
            let previous = load_raw_frag_reuse_content(&stale, &current_entry)
                .await
                .unwrap();
            let trimmed = load_raw_frag_reuse_content(&stale, &trimmed_entry)
                .await
                .unwrap();
            assert!(previous.ends_with(&trimmed));
            assert!(trimmed.len() < previous.len());
        }

        /// Trim vs. trim mirrors the v0 cleanup-vs-cleanup conflict.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn concurrent_trims_conflict() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 20).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("i_idx".into()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();

            let mut stale = dataset.clone();
            cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            assert!(fri_entry(&dataset).await.is_none());
            assert!(matches!(
                cleanup_frag_reuse_index(&mut stale).await,
                Err(Error::RetryableCommitConflict { .. })
            ));
        }

        /// Trim vs. tagged compaction: the freshly appended
        /// ordered-compaction transition survives the rebased trim.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn trim_rebases_over_concurrent_tagged_compaction() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 40).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("i_idx".into()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();

            let mut stale = dataset.clone();
            // A tagged compaction appends an ordered-compaction transition.
            compact_files(
                &mut dataset,
                CompactionOptions {
                    target_rows_per_fragment: 100,
                    defer_index_remap: true,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            let ledger = decode_frag_reuse_ledger(&dataset, &fri_entry(&dataset).await.unwrap())
                .await
                .unwrap();
            assert_eq!(ledger.transitions().len(), 2);

            cleanup_frag_reuse_index(&mut stale).await.unwrap();
            let trimmed_entry = fri_entry(&stale).await.unwrap();
            let ledger = decode_frag_reuse_ledger(&stale, &trimmed_entry)
                .await
                .unwrap();
            assert_eq!(ledger.transitions().len(), 1);
            assert!(matches!(
                ledger.transitions()[0].mapping(),
                Mapping::OrderedCompaction(_)
            ));
            assert_eq!(
                ledger.transitions()[0]
                    .sources()
                    .iter()
                    .map(|d| d.id)
                    .collect::<Vec<_>>(),
                vec![10, 11]
            );
        }

        /// Commit a fresh `i_idx` delta segment built over the current
        /// (translated) table state. `covered` narrows the committed bitmap
        /// to an under-claim, standing in for a segment that directly covers
        /// only part of the destinations.
        async fn commit_delta_segment(
            dataset: &mut Dataset,
            covered: Option<&[u32]>,
        ) -> uuid::Uuid {
            let params = ScalarIndexParams::default();
            let mut delta =
                crate::index::CreateIndexBuilder::new(dataset, &["i"], IndexType::BTree, &params)
                    .name("i_idx_delta".into())
                    .execute_uncommitted()
                    .await
                    .unwrap();
            delta.name = "i_idx".into();
            if let Some(covered) = covered {
                delta.fragment_bitmap = Some(covered.iter().copied().collect());
            }
            let uuid = delta.uuid;
            dataset
                .apply_commit(
                    Transaction::new(
                        dataset.manifest.version,
                        Operation::CreateIndex {
                            new_indices: vec![delta],
                            removed_indices: vec![],
                        },
                        None,
                    ),
                    &Default::default(),
                    &Default::default(),
                )
                .await
                .unwrap();
            uuid
        }

        async fn stored_segments(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
            read_manifest_indexes(
                &dataset.object_store,
                &dataset.manifest_location,
                &dataset.manifest,
            )
            .await
            .unwrap()
            .into_iter()
            .filter(|idx| idx.name == name)
            .collect()
        }

        /// Full takeover: a delta directly covering everything the old
        /// segment reaches through the chain supersedes it. The prune runs
        /// in the same maintenance invocation as the trim, which then
        /// releases the transition only the pruned segment needed -- and
        /// query results are identical before and after (direct-coverage-wins
        /// already masked the pruned segment's contribution).
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn superseded_segment_pruned_then_transition_released() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 20).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            let old_segment = &stored_segments(&dataset, "i_idx").await[0].clone();
            let delta = commit_delta_segment(&mut dataset, None).await;
            assert_eq!(stored_segments(&dataset, "i_idx").await.len(), 2);

            let before_all = sorted_values(&dataset).await;
            let before_filtered = dataset.count_rows(Some("i >= 4".into())).await.unwrap();

            cleanup_frag_reuse_index(&mut dataset).await.unwrap();

            // The old segment is pruned, the delta survives...
            let segments = stored_segments(&dataset, "i_idx").await;
            assert_eq!(
                segments.iter().map(|s| s.uuid).collect::<Vec<_>>(),
                vec![delta]
            );
            assert_ne!(segments[0].uuid, old_segment.uuid);
            // ... and the same invocation's trim released the transition
            // only the pruned segment needed (the entry is fully drained).
            assert!(fri_entry(&dataset).await.is_none());
            // Safety: pruning changed no query result.
            assert_eq!(sorted_values(&dataset).await, before_all);
            assert_eq!(
                dataset.count_rows(Some("i >= 4".into())).await.unwrap(),
                before_filtered
            );
        }

        /// A segment still directly covering a live fragment nobody else
        /// covers is NOT superseded, even with all its translated
        /// destinations taken over.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn live_direct_coverage_prevents_pruning() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 20).await;
            // Only fragment 1 is repartitioned; the old segment keeps live
            // direct coverage of fragment 0.
            let mut dataset = commit_stable_partition(dataset, &[1], 10).await;
            commit_delta_segment(&mut dataset, Some(&[10, 11])).await;

            assert!(
                derive_superseded_segments(&dataset)
                    .await
                    .unwrap()
                    .is_empty(),
                "a segment with exclusive live direct coverage must be kept"
            );
            cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            assert_eq!(stored_segments(&dataset, "i_idx").await.len(), 2);
        }

        /// Partial takeover: one destination still lacks direct coverage, so
        /// the old segment stays (and so does the transition).
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn partial_takeover_keeps_segment_and_transition() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 20).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            commit_delta_segment(&mut dataset, Some(&[10])).await;

            assert!(
                derive_superseded_segments(&dataset)
                    .await
                    .unwrap()
                    .is_empty()
            );
            cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            assert_eq!(stored_segments(&dataset, "i_idx").await.len(), 2);
            assert!(
                fri_entry(&dataset).await.is_some(),
                "destination 11 lacks direct coverage, so the transition stays"
            );
        }

        /// Mutual redundancy: of segments fully covering each other, exactly
        /// one survives, deterministically the newest.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn mutual_redundancy_keeps_single_newest_segment() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 20).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            let _delta1 = commit_delta_segment(&mut dataset, None).await;
            let delta2 = commit_delta_segment(&mut dataset, None).await;
            assert_eq!(stored_segments(&dataset, "i_idx").await.len(), 3);

            let before = sorted_values(&dataset).await;
            cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            let segments = stored_segments(&dataset, "i_idx").await;
            assert_eq!(
                segments.iter().map(|s| s.uuid).collect::<Vec<_>>(),
                vec![delta2],
                "exactly the newest of the mutually redundant segments survives"
            );
            assert_eq!(sorted_values(&dataset).await, before);
        }

        /// An unknown envelope-level record may carry lineage this build
        /// cannot see: both the splice parser and the ledger refuse, so the
        /// trim never rewrites such a history.
        #[tokio::test]
        async fn unknown_envelope_record_refuses_trim() {
            use prost::Message;
            let transition = pb_fri::Transition {
                sources: vec![digest(1)],
                destinations: vec![digest(2)],
                mapping: Some(pb_fri::transition::Mapping::StablePartition(
                    pb_fri::StablePartition {
                        map_id: Uuid::new_v4().to_string(),
                        map_size_bytes: 1,
                        base_id: None,
                    },
                )),
            };
            let mut content = pb_fri::InlineContent {
                legacy_versions: vec![],
                transitions: vec![transition],
            }
            .encode_to_vec();
            prost::encoding::encode_key(
                9,
                prost::encoding::WireType::LengthDelimited,
                &mut content,
            );
            prost::encoding::encode_varint(6, &mut content);
            content.extend_from_slice(b"future");

            let Err(error) = split_trim_elements(&content) else {
                panic!("an unknown envelope record must refuse the splice")
            };
            assert!(matches!(error, Error::NotSupported { .. }), "{error}");
            let ledger =
                crate::index::frag_reuse::decode_frag_reuse_ledger_from_content(1, &content)
                    .await
                    .unwrap();
            assert!(
                ledger.has_unsupported_transitions(),
                "the ledger must report envelope-level unknowns so every \
                 maintenance path (and _fri GC) refuses consistently"
            );
        }

        fn unusable_details() -> Option<Arc<prost_types::Any>> {
            Some(Arc::new(prost_types::Any {
                type_url: "/lance.index.FutureIndexDetails".into(),
                value: vec![],
            }))
        }

        /// Direct coverage from a segment this build cannot serve must not
        /// mask a transition: only usable segments count as taking over a
        /// destination, while any segment's provenance still pins.
        #[test]
        fn unusable_segments_do_not_mask_transition_coverage() {
            let elements = vec![transition_element(&[1, 2], &[5, 6])];
            let pinner = named_index("a_idx", 1, &[1, 2]);
            let mut unusable_taker = named_index("a_idx", 2, &[5, 6]);
            unusable_taker.index_details = unusable_details();
            assert_eq!(
                compute_tagged_retention(&elements, &[pinner.clone(), unusable_taker]),
                vec![true],
                "an unusable segment's bitmap must not stand in as direct coverage"
            );
            // The same takeover by a usable segment releases the transition.
            assert_eq!(
                compute_tagged_retention(&elements, &[pinner, named_index("a_idx", 2, &[5, 6])]),
                vec![false]
            );
        }

        /// Integration: an unusable segment claiming the destinations
        /// neither prunes the usable old segment nor releases the
        /// transition.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn unusable_segment_neither_masks_nor_prunes() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 20).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            let old_segment = stored_segments(&dataset, "i_idx").await[0].clone();

            // A segment of a type this build has no reader for, claiming the
            // destinations directly (e.g. written by a newer Lance).
            let unusable = IndexMetadata {
                uuid: Uuid::new_v4(),
                fields: old_segment.fields.clone(),
                covering_fields: vec![],
                name: "i_idx".into(),
                dataset_version: dataset.manifest.version,
                fragment_bitmap: Some([10u32, 11].into_iter().collect()),
                index_details: unusable_details(),
                index_version: 0,
                created_at: None,
                base_id: None,
                files: None,
            };
            dataset
                .apply_commit(
                    Transaction::new(
                        dataset.manifest.version,
                        Operation::CreateIndex {
                            new_indices: vec![unusable],
                            removed_indices: vec![],
                        },
                        None,
                    ),
                    &Default::default(),
                    &Default::default(),
                )
                .await
                .unwrap();

            cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            let segments = stored_segments(&dataset, "i_idx").await;
            assert!(
                segments.iter().any(|s| s.uuid == old_segment.uuid),
                "the only segment this build can serve must not be pruned"
            );
            assert_eq!(segments.len(), 2, "the unusable segment is kept too");
            assert!(
                fri_entry(&dataset).await.is_some(),
                "a destination covered only by an unusable segment keeps its transition"
            );
        }

        async fn narrow_index_bitmap(dataset: &mut Dataset, name: &str, covered: &[u32]) {
            let original = stored_segments(dataset, name).await[0].clone();
            let mut narrowed = original.clone();
            narrowed.fragment_bitmap = Some(covered.iter().copied().collect());
            dataset
                .apply_commit(
                    Transaction::new(
                        dataset.manifest.version,
                        Operation::CreateIndex {
                            new_indices: vec![narrowed],
                            removed_indices: vec![original],
                        },
                        None,
                    ),
                    &Default::default(),
                    &Default::default(),
                )
                .await
                .unwrap();
        }

        /// A trim whose rebase finds nothing left to trim (a concurrent
        /// commit re-pinned the history mid-trim) must abort instead of
        /// writing an empty no-op version.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn rebased_noop_trim_commits_no_version() {
            let mut dataset = reader_tests::fixture().await;
            reserve_fragments(&mut dataset, 20).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            // Drain: the trim would delete the entry.
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("i_idx".into()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();

            let mut stale = dataset.clone();
            // Concurrent commit re-pins the history: the segment's coverage
            // returns to the sources, so nothing is trimmable anymore.
            narrow_index_bitmap(&mut dataset, "i_idx", &[0, 1]).await;
            let expected_version = dataset.manifest.version;

            cleanup_frag_reuse_index(&mut stale).await.unwrap();
            assert_eq!(
                stale.manifest.version, expected_version,
                "the rebased no-op trim must not write a version"
            );
            assert!(
                fri_entry(&stale).await.is_some(),
                "the re-pinned history stays"
            );
        }

        /// v0 guard: pruning only runs in the tagged branch; a v0 cleanup
        /// leaves redundant segments alone.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn v0_cleanup_never_prunes_segments() {
            let mut dataset = reader_tests::fixture().await;
            commit_delta_segment(&mut dataset, None).await;
            assert_eq!(stored_segments(&dataset, "i_idx").await.len(), 2);

            // A v0 FRI entry via deferred compaction on the untagged table.
            dataset.delete("i < 2").await.unwrap();
            compact_files(
                &mut dataset,
                CompactionOptions {
                    target_rows_per_fragment: 100,
                    defer_index_remap: true,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            let entry = fri_entry(&dataset).await.unwrap();
            assert_eq!(entry.index_version, 0);

            cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            assert_eq!(
                stored_segments(&dataset, "i_idx").await.len(),
                2,
                "the v0 path must not prune segments"
            );
        }
    }

    async fn row_addrs_by_i(dataset: &Dataset) -> HashMap<i32, u64> {
        let batch = dataset
            .scan()
            .project(&["i"])
            .unwrap()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let ids = batch["i"].as_primitive::<Int32Type>();
        let addrs = batch[ROW_ADDR].as_primitive::<arrow_array::types::UInt64Type>();
        ids.values()
            .iter()
            .copied()
            .zip(addrs.values().iter().copied())
            .collect()
    }

    #[tokio::test]
    async fn test_frag_reuse_index_accessor() {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        assert!(dataset.frag_reuse_index().await.unwrap().is_none());

        // Non-deferred compaction of fragment 0 (deletions above the threshold) records no FRI.
        dataset.delete("i < 200").await.unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1_000,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(dataset.frag_reuse_index().await.unwrap().is_none());
        let num_fragments = dataset.fragments().len();
        assert_eq!(num_fragments, 6);

        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("i_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Deletions above the threshold make exactly these two fragments candidates.
        dataset.delete("i >= 1000 AND i < 1250").await.unwrap();
        dataset.delete("i >= 2000 AND i < 2250").await.unwrap();
        let before = row_addrs_by_i(&dataset).await;
        let pre_compaction_version = dataset.version().version;
        let rewritten_frags = [
            RowAddress::from(before[&1250]).fragment_id(),
            RowAddress::from(before[&2250]).fragment_id(),
        ];
        let untouched_addr = before[&5000];
        // Offset 0 of the first rewritten fragment is i=1000, deleted above.
        let deleted_addr = u64::from(RowAddress::new_from_parts(rewritten_frags[0], 0));

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let after = row_addrs_by_i(&dataset).await;
        for frag in rewritten_frags {
            assert!(
                dataset.fragments().iter().all(|f| f.id != u64::from(frag)),
                "fragment {frag} should have been rewritten"
            );
        }

        let frag_reuse_index = dataset
            .frag_reuse_index()
            .await
            .unwrap()
            .expect("deferred compaction must record an FRI");
        let frag_reuse_index_meta = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .expect("FRI must be in the manifest");
        assert_eq!(frag_reuse_index.uuid, frag_reuse_index_meta.uuid);
        assert_eq!(frag_reuse_index.details.versions.len(), 1);
        assert_false!(frag_reuse_index.is_empty());

        // Cache reuse, not an API guarantee.
        let again = dataset.frag_reuse_index().await.unwrap().unwrap();
        assert!(Arc::ptr_eq(&frag_reuse_index, &again));

        let remap = frag_reuse_index.row_addr_remap();
        for i in [1250, 1999, 2250, 2999] {
            assert_eq!(
                remap.get(before[&i]),
                Some(Some(after[&i])),
                "row i={i} should have moved"
            );
            assert_eq!(frag_reuse_index.remap_row_id(before[&i]), Some(after[&i]));
        }
        assert_eq!(remap.get(deleted_addr), Some(None));
        assert_eq!(frag_reuse_index.remap_row_id(deleted_addr), None);
        assert_eq!(remap.get(untouched_addr), None);
        assert_eq!(after[&5000], untouched_addr);
        assert_eq!(
            frag_reuse_index.remap_row_id(untouched_addr),
            Some(untouched_addr)
        );

        let pre_compaction = dataset
            .checkout_version(pre_compaction_version)
            .await
            .unwrap();
        assert!(pre_compaction.frag_reuse_index().await.unwrap().is_none());

        remapping::remap_column_index(&mut dataset, &["i"], Some("i_idx".into()))
            .await
            .unwrap();
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();
        let trimmed = dataset
            .frag_reuse_index()
            .await
            .unwrap()
            .expect("trimmed FRI keeps an (empty) manifest entry");
        assert_true!(trimmed.is_empty());
        assert_eq!(trimmed.details.versions.len(), 0);
        assert_ne!(trimmed.uuid, frag_reuse_index.uuid);
        assert_eq!(trimmed.row_addr_remap().get(before[&1250]), None);
        assert_eq!(
            trimmed.remap_row_id(before[&1250]),
            Some(before[&1250]),
            "trimmed history passes a moved address through unchanged"
        );
    }

    /// Deleting every row of a destination fragment removes it from the manifest,
    /// but the FRI still maps into it.
    #[tokio::test]
    async fn test_frag_reuse_index_mapped_destination_can_be_removed() {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(1000))
            .await
            .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("i_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Target equals fragment size, so only the two deletion-heavy fragments are candidates.
        dataset.delete("i < 250").await.unwrap();
        dataset.delete("i >= 1000 AND i < 1250").await.unwrap();
        let before = row_addrs_by_i(&dataset).await;
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let after = row_addrs_by_i(&dataset).await;
        let moved_rows = [250, 999, 1250, 1999];
        let destination_frags: Vec<u32> = moved_rows
            .iter()
            .map(|i| RowAddress::from(after[i]).fragment_id())
            .collect();
        for i in moved_rows {
            assert_ne!(before[&i], after[&i], "row i={i} should have moved");
        }

        dataset.delete("i < 2000").await.unwrap();
        let remaining_frag_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
        for frag in &destination_frags {
            assert!(
                !remaining_frag_ids.contains(&u64::from(*frag)),
                "destination fragment {frag} should be removed; remaining: {remaining_frag_ids:?}"
            );
        }

        let frag_reuse_index = dataset.frag_reuse_index().await.unwrap().unwrap();
        assert_eq!(frag_reuse_index.details.versions.len(), 1);
        for i in moved_rows {
            assert_eq!(
                frag_reuse_index.row_addr_remap().get(before[&i]),
                Some(Some(after[&i])),
                "row i={i} still maps into a removed fragment"
            );
            assert_eq!(frag_reuse_index.remap_row_id(before[&i]), Some(after[&i]));
        }
    }
}
