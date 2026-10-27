// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Settling what an operation asks of the fragment reuse history.
//!
//! [`AppendFragReuseTransition`](super::super::AppendFragReuseTransition) and
//! [`TrimFragReuse`](super::super::TrimFragReuse) only stage their request as
//! they apply: a transition's sources and destinations may be removed and added
//! by actions anywhere in the operation, so it is validated once the whole
//! operation has run. The history is then rebuilt from the one the version
//! being committed on top of carries, never from a copy the writer read, which
//! is what makes a retry merge onto whatever a concurrent writer recorded.

use super::ApplyState;
use crate::feature_flags::{FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS, FLAG_FRAGMENT_REUSE_INDEX};
use crate::format::pb::fragment_reuse_index_details as pb_fri;
use crate::format::{Fragment, IndexMetadata, Manifest};
use crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use crate::system_index::frag_reuse::ledger::{FragReuseLedger, Mapping, content_field};
use crate::system_index::frag_reuse::lineage::TaggedLineage;
use crate::system_index::frag_reuse::metadata::is_tagged;
use crate::system_index::frag_reuse::retention::{compute_tagged_retention, split_trim_elements};
use crate::system_index::is_system_index;
use crate::transaction::Transaction;
use crate::transaction::action::FragReuseMapping;
use bytes::Bytes;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use roaring::{RoaringBitmap, RoaringTreemap};
use std::collections::HashSet;
use std::io::Cursor;
use std::sync::Arc;
use uuid::Uuid;

/// The most destinations a stable partition can label: a label is a `uint16`.
const MAX_STABLE_PARTITION_DESTINATIONS: usize = 1 << 16;

const DETAILS_TYPE_URL: &str = "/lance.table.FragmentReuseIndexDetails";

/// A transition as an action staged it, with its fragments resolved.
#[derive(Debug)]
struct StagedTransition {
    sources: Vec<u64>,
    destinations: Vec<u64>,
    mapping: FragReuseMapping,
}

/// What an operation asks of the fragment reuse history.
#[derive(Debug, Default)]
pub(super) struct FragReuseChanges {
    transitions: Vec<StagedTransition>,
    trim: bool,
}

impl FragReuseChanges {
    fn is_empty(&self) -> bool {
        self.transitions.is_empty() && !self.trim
    }
}

/// Mark a manifest that publishes a fragment reuse history: the tagged form
/// needs both FRI capability bits, and a table with stable row ids also
/// declares that it carries both.
pub(super) fn set_feature_flags(manifest: &mut Manifest, indices: &[IndexMetadata]) {
    let Some(entry) = indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
    else {
        return;
    };
    let mut flags = 0;
    if is_tagged(entry) {
        flags |= FLAG_FRAGMENT_REUSE_INDEX;
    }
    if manifest.uses_stable_row_ids() {
        flags |= FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
    }
    manifest.reader_feature_flags |= flags;
    manifest.writer_feature_flags |= flags;
}

impl ApplyState<'_> {
    pub(in crate::transaction::action) fn stage_frag_reuse_transition(
        &mut self,
        sources: Vec<u64>,
        destinations: Vec<u64>,
        mapping: FragReuseMapping,
    ) {
        self.frag_reuse.transitions.push(StagedTransition {
            sources,
            destinations,
            mapping,
        });
    }

    pub(in crate::transaction::action) fn request_frag_reuse_trim(&mut self) {
        self.frag_reuse.trim = true;
    }

    /// Whether the table carries a tagged fragment reuse history.
    pub(in crate::transaction::action) fn carries_tagged_history(&self) -> bool {
        self.indices.iter().any(is_tagged)
    }

    /// Append the staged transitions to the current history, trim it if
    /// asked, and withdraw coverage an in-place rewrite invalidated along the
    /// lineage. Whether the operation published a changed history.
    pub(super) fn finish_frag_reuse_history(&mut self) -> Result<bool> {
        let changes = std::mem::take(&mut self.frag_reuse);
        let rebound_fields = std::mem::take(&mut self.rebound_fields);
        let position = self
            .indices
            .iter()
            .position(|index| index.name == FRAG_REUSE_INDEX_NAME);

        if changes.is_empty() {
            if let Some(position) = position
                && is_tagged(&self.indices[position])
                && !rebound_fields.is_empty()
            {
                let entry = self.indices[position].clone();
                let ledger = self.decode_history(&entry, &self.history_content(&entry)?)?;
                self.withdraw_along_lineage(&entry, ledger, rebound_fields);
            }
            return Ok(false);
        }

        let current = position.map(|position| self.indices[position].clone());
        let (mut content, mut index_version) = match &current {
            Some(entry) => {
                if !matches!(entry.index_version, 0 | 1) {
                    return Err(Error::not_supported(format!(
                        "the fragment reuse history is at index_version {}, which this version of \
                         Lance cannot maintain; upgrade to append to or trim it",
                        entry.index_version
                    )));
                }
                let content = self.history_content(entry)?;
                let ledger = self.decode_history(entry, &content)?;
                if ledger.has_unsupported_transitions() {
                    return Err(Error::not_supported(
                        "the fragment reuse history carries records this version of Lance cannot \
                         interpret, so it cannot maintain it; upgrade to append to or trim it",
                    ));
                }
                self.check_transitions(&changes.transitions, Some(&ledger))?;
                (content.to_vec(), entry.index_version)
            }
            None => {
                self.check_transitions(&changes.transitions, None)?;
                (Vec::new(), 1)
            }
        };

        for staged in &changes.transitions {
            let transition = self.recorded_transition(staged)?;
            prost::encoding::message::encode(2, &transition, &mut content);
            index_version = 1;
        }

        if changes.trim && current.is_some() {
            let elements = split_trim_elements(&content)?;
            let retained = compute_tagged_retention(
                &elements,
                &self.indices,
                self.frag_reuse_inputs.index_is_usable,
            );
            if retained.iter().any(|kept| !kept) {
                content = elements
                    .iter()
                    .zip(retained)
                    .filter(|(_, kept)| *kept)
                    .flat_map(|(element, _)| element.raw.iter().copied())
                    .collect();
            } else if changes.transitions.is_empty() {
                // Nothing to trim and nothing appended: the entry stands.
                return Ok(false);
            }
        }

        if let Some(position) = position {
            self.indices.remove(position);
        }
        if content.is_empty() {
            // Trimmed to nothing. The capability flags are sticky, so the table
            // keeps the tagged record form.
            return Ok(current.is_some());
        }

        let content = Bytes::from(content);
        let ledger = Arc::new(
            FragReuseLedger::decode_content(index_version, content.clone()).map_err(|error| {
                Error::invalid_input(format!(
                    "the fragment reuse history this operation assembles is invalid: {error}"
                ))
            })?,
        );
        check_row_maps_are_owned(&ledger)?;
        let entry = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: FRAG_REUSE_INDEX_NAME.to_string(),
            fields: Vec::new(),
            covering_fields: Vec::new(),
            dataset_version: self.current_manifest.version,
            fragment_bitmap: Some(lineage_fragments(&ledger)),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: DETAILS_TYPE_URL.into(),
                value: {
                    let mut value = Vec::new();
                    prost::encoding::bytes::encode(1, &content, &mut value);
                    value
                },
            })),
            index_version,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            // A stable partition's row map lives in its own directory, named by
            // the transition, not under this entry's uuid.
            files: None,
        };
        if is_tagged(&entry) && !rebound_fields.is_empty() {
            self.withdraw_along_lineage(&entry, ledger, rebound_fields);
        }
        self.indices.push(entry);
        Ok(true)
    }

    /// The serialized `InlineContent` of `entry`: from the entry itself when
    /// it is inline, else as the commit path read it ahead of the build.
    fn history_content(&self, entry: &IndexMetadata) -> Result<Bytes> {
        let details = entry.index_details.as_ref().ok_or_else(|| {
            Error::invalid_input(
                "the fragment reuse entry carries no details, so its history cannot be read",
            )
        })?;
        let (tag, content) = content_field(details)?;
        if tag == 1 {
            return Ok(content);
        }
        match &self.frag_reuse_inputs.history {
            Some(preloaded) if preloaded.entry == entry.uuid => Ok(preloaded.content.clone()),
            Some(preloaded) => Err(Error::invalid_input(format!(
                "the fragment reuse history was read from entry {}, but the version being \
                 committed on top of carries entry {}; read it again for this attempt",
                preloaded.entry, entry.uuid
            ))),
            None => Err(Error::invalid_input(
                "the fragment reuse history is stored externally and was not read ahead of the \
                 manifest build; commit through the dataset so the history is read first",
            )),
        }
    }

    fn decode_history(
        &self,
        entry: &IndexMetadata,
        content: &Bytes,
    ) -> Result<Arc<FragReuseLedger>> {
        FragReuseLedger::decode_content(entry.index_version, content.clone()).map(Arc::new)
    }

    /// Withdraw coverage that segments translating into a fragment rewritten
    /// in place reach through `ledger`. The rewritten fragment itself was
    /// already pruned as the tombstone applied.
    fn withdraw_along_lineage(
        &mut self,
        entry: &IndexMetadata,
        ledger: Arc<FragReuseLedger>,
        rebound_fields: Vec<(u64, Vec<u32>)>,
    ) {
        let lineage = TaggedLineage::new(entry, Some(ledger));
        let live: RoaringBitmap = self
            .fragments
            .iter()
            .filter_map(|fragment| u32::try_from(fragment.id).ok())
            .collect();
        Transaction::withdraw_in_place_rewrites(
            &mut self.indices,
            rebound_fields,
            &self.schema,
            &live,
            Some(&lineage),
        );
    }

    /// The checks that need the whole operation: what removed the sources and
    /// added the destinations, and what else the history and the index list
    /// hold.
    fn check_transitions(
        &self,
        transitions: &[StagedTransition],
        ledger: Option<&FragReuseLedger>,
    ) -> Result<()> {
        if transitions.is_empty() {
            return Ok(());
        }
        if self.uses_stable_row_ids()
            && let Some(index) = self
                .indices
                .iter()
                .find(|index| !is_system_index(index) && !index.results_are_row_addrs())
        {
            return Err(Error::invalid_input(format!(
                "AppendFragReuseTransition cannot record row moves on this table: index '{}' \
                 (segment {}) stores stable row ids rather than row addresses, and the fragment \
                 reuse history translates row addresses",
                index.name, index.uuid
            )));
        }
        let mut consumed = HashSet::new();
        let mut produced = HashSet::new();
        for transition in transitions {
            if matches!(transition.mapping, FragReuseMapping::StablePartition(_))
                && !(1..=MAX_STABLE_PARTITION_DESTINATIONS).contains(&transition.destinations.len())
            {
                return Err(Error::invalid_input(format!(
                    "AppendFragReuseTransition with a stable partition names {} destinations; a \
                     stable partition labels between 1 and {MAX_STABLE_PARTITION_DESTINATIONS}",
                    transition.destinations.len()
                )));
            }
            for &source in &transition.sources {
                match self.removed_fragments.get(&source) {
                    Some(false) => {}
                    Some(true) => {
                        return Err(Error::invalid_input(format!(
                            "AppendFragReuseTransition consumes fragment {source}, which this \
                             operation removes with data_change true; a transition records rows \
                             moved without changing their values, so its sources are removed \
                             with data_change false"
                        )));
                    }
                    None => {
                        return Err(Error::invalid_input(format!(
                            "AppendFragReuseTransition consumes fragment {source}, which this \
                             operation does not remove; remove each source with RemoveFragment"
                        )));
                    }
                }
                let already_consumed = ledger.is_some_and(|ledger| {
                    u32::try_from(source)
                        .ok()
                        .and_then(|source| ledger.consumer(source))
                        .is_some()
                });
                if already_consumed || !consumed.insert(source) {
                    return Err(Error::invalid_input(format!(
                        "AppendFragReuseTransition consumes fragment {source}, which another \
                         transition of the fragment reuse history already consumes"
                    )));
                }
            }
            for &destination in &transition.destinations {
                if !self.is_new_fragment(destination) {
                    return Err(Error::invalid_input(format!(
                        "AppendFragReuseTransition produces fragment {destination}, which this \
                         operation does not add; add each destination with AddFragment"
                    )));
                }
                if !produced.insert(destination) {
                    return Err(Error::invalid_input(format!(
                        "AppendFragReuseTransition produces fragment {destination} more than once"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The transition as the history records it, with digests derived from
    /// the fragments and the mapping checked to conserve rows.
    fn recorded_transition(&self, staged: &StagedTransition) -> Result<pb_fri::Transition> {
        let sources = staged
            .sources
            .iter()
            .map(|&id| {
                let fragment = self
                    .current_manifest
                    .fragments
                    .iter()
                    .find(|fragment| fragment.id == id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "AppendFragReuseTransition consumes fragment {id}, which the version \
                             being committed on top of does not have"
                        ))
                    })?;
                source_digest(fragment)
            })
            .collect::<Result<Vec<_>>>()?;
        let destinations = staged
            .destinations
            .iter()
            .map(|&id| {
                let fragment = self
                    .fragments
                    .iter()
                    .find(|fragment| fragment.id == id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "AppendFragReuseTransition produces fragment {id}, which the \
                             operation removes again"
                        ))
                    })?;
                destination_digest(fragment)
            })
            .collect::<Result<Vec<_>>>()?;

        let surviving: u64 = sources
            .iter()
            .map(|digest| digest.physical_rows - digest.num_deleted_rows)
            .sum();
        let written: u64 = destinations.iter().map(|digest| digest.physical_rows).sum();
        if surviving != written {
            return Err(Error::invalid_input(format!(
                "AppendFragReuseTransition does not conserve rows: sources {:?} hold {surviving} \
                 surviving rows, destinations {:?} hold {written}",
                staged.sources, staged.destinations
            )));
        }
        if let FragReuseMapping::OrderedCompaction(ordered) = &staged.mapping {
            check_ordered_compaction(ordered, &sources)?;
        }

        Ok(pb_fri::Transition {
            sources,
            destinations,
            mapping: Some(staged.mapping.clone()),
        })
    }
}

fn source_digest(fragment: &Fragment) -> Result<pb_fri::FragmentDigest> {
    // Materializing an overlay would move overlaid values into the
    // destination while a segment older than the overlay kept translating
    // its stale entries there.
    if !fragment.overlays.is_empty() {
        return Err(Error::not_supported(format!(
            "AppendFragReuseTransition consumes fragment {}, which carries data overlay files; \
             compact the overlays away or rebuild the covering indices before recording the \
             move",
            fragment.id
        )));
    }
    let physical_rows = fragment.physical_rows.ok_or_else(|| {
        Error::invalid_input(format!(
            "AppendFragReuseTransition consumes fragment {}, which records no physical row count",
            fragment.id
        ))
    })? as u64;
    let num_deleted_rows = match &fragment.deletion_file {
        None => 0,
        Some(deletion_file) => deletion_file.num_deleted_rows.ok_or_else(|| {
            Error::invalid_input(format!(
                "AppendFragReuseTransition consumes fragment {}, whose deletion file records no \
                 deleted row count; apply cannot read the file to count it",
                fragment.id
            ))
        })? as u64,
    };
    Ok(pb_fri::FragmentDigest {
        id: fragment.id,
        physical_rows,
        num_deleted_rows,
    })
}

fn destination_digest(fragment: &Fragment) -> Result<pb_fri::FragmentDigest> {
    if fragment.deletion_file.is_some() {
        return Err(Error::invalid_input(format!(
            "AppendFragReuseTransition produces fragment {}, which carries a deletion file; a \
             destination holds exactly the rows moved into it",
            fragment.id
        )));
    }
    let physical_rows = fragment.physical_rows.ok_or_else(|| {
        Error::invalid_input(format!(
            "AppendFragReuseTransition produces fragment {}, which records no physical row count",
            fragment.id
        ))
    })? as u64;
    Ok(pb_fri::FragmentDigest {
        id: fragment.id,
        physical_rows,
        num_deleted_rows: 0,
    })
}

/// Each source's surviving rows, as the bitmap marks them, must be exactly
/// the rows the fragment has not deleted, and the bitmap must mark nothing
/// outside the sources.
fn check_ordered_compaction(
    ordered: &pb_fri::OrderedCompaction,
    sources: &[pb_fri::FragmentDigest],
) -> Result<()> {
    let mut cursor = Cursor::new(&ordered.changed_row_addrs);
    let bitmap = RoaringTreemap::deserialize_from(&mut cursor).map_err(|error| {
        Error::invalid_input(format!(
            "AppendFragReuseTransition carries an ordered compaction whose surviving-row bitmap \
             does not decode: {error}"
        ))
    })?;
    let mut marked = 0;
    for source in sources {
        let fragment = u32::try_from(source.id).map_err(|_| {
            Error::invalid_input(format!(
                "AppendFragReuseTransition consumes fragment {}, beyond the row-address range",
                source.id
            ))
        })?;
        let survivors = bitmap.range_cardinality(RowAddress::address_range(fragment));
        let expected = source.physical_rows - source.num_deleted_rows;
        if survivors != expected {
            return Err(Error::invalid_input(format!(
                "AppendFragReuseTransition marks {survivors} surviving rows in fragment {}, which \
                 has {} physical rows of which {} are deleted",
                source.id, source.physical_rows, source.num_deleted_rows
            )));
        }
        marked += survivors;
    }
    if marked != bitmap.len() {
        return Err(Error::invalid_input(format!(
            "AppendFragReuseTransition marks {} surviving rows, {} of them outside its sources",
            bitmap.len(),
            bitmap.len() - marked
        )));
    }
    Ok(())
}

/// Each stable partition owns its row map: maintenance of one transition
/// must never delete or overwrite a map another still references.
fn check_row_maps_are_owned(ledger: &FragReuseLedger) -> Result<()> {
    let mut seen = HashSet::new();
    for transition in ledger.transitions() {
        if let Mapping::StablePartition(partition) = transition.mapping()
            && !seen.insert(partition.map_id.as_str())
        {
            return Err(Error::invalid_input(format!(
                "stable-partition row map {} is referenced by more than one transition; each \
                 transition must own its row map",
                partition.map_id
            )));
        }
    }
    Ok(())
}

/// Every fragment the history's records touch, sources and destinations.
fn lineage_fragments(ledger: &FragReuseLedger) -> RoaringBitmap {
    ledger
        .transitions()
        .iter()
        .flat_map(|transition| transition.sources().iter().chain(transition.destinations()))
        .filter_map(|digest| u32::try_from(digest.id).ok())
        .collect()
}
