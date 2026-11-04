// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The tagged (`index_version` 1) fragment reuse history: compaction versions
//! and tagged transitions. Every writer of this format validates it, so a
//! tagged history is never repaired.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use bytes::{Buf, Bytes, BytesMut};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_select::Remapper;
use lance_select::remapper::{RemapperBacking, TranslateFuture};
use prost::Message;
use prost::encoding::{WireType, decode_key, decode_varint};
use roaring::RoaringBitmap;

use super::{FriImpl, Records, StoredRecord, concat, invalid_edit};
use crate::format::pb::fragment_reuse_index_details::{self as pb, transition};
use crate::system_index::frag_reuse::history::{
    HistoryFormat, Record, RewriteRecord, TRANSITION_FIELD, length_delimited,
};
use crate::system_index::frag_reuse::ledger::{
    FragReuseLedger, Mapping, STABLE_PARTITION_FIELD, corrupt, next_field, require_message,
};
use crate::system_index::frag_reuse::mapping::{
    MappingRef, MappingSource, translate_stable_partition,
};
use crate::system_index::frag_reuse::plan::{
    Composition, ReadContext, ReadPlan, RewriteContext, SegmentSummary, TranslationPlan,
    TranslationSupport, segment_plans,
};

mod compose;
mod trim;

/// `StablePartition.base_id`.
const STABLE_PARTITION_BASE_ID_FIELD: u32 = 3;

#[derive(Debug, Default)]
pub struct TaggedHistory(Records);

impl TaggedHistory {
    pub fn decode(content: Bytes) -> Result<Self> {
        let records = Records::decode(HistoryFormat::Tagged, content)?;
        validate_records(&records.stored)?;
        Ok(Self(records))
    }
}

impl TaggedHistory {
    /// Whether the segment's stored addresses are usable as they are. A
    /// fragment the segment covers that is no longer live needs no
    /// translation either: the reader serves only its live coverage.
    fn is_direct(&self, context: &ReadContext, segment: &SegmentSummary) -> bool {
        self.is_caught_up(context.entry_version, segment)
    }
}

impl FriImpl for TaggedHistory {
    fn format(&self) -> HistoryFormat {
        HistoryFormat::Tagged
    }

    fn records(&self) -> &Records {
        &self.0
    }

    fn rebuilt(&self, content: Bytes) -> Result<Arc<dyn FriImpl>> {
        Ok(Arc::new(Self::decode(content)?))
    }

    /// Every fragment of the retained lineage, sources included, so a rewrite
    /// of a fragment the history still translates from is seen as covered.
    fn fragment_bitmap(&self) -> RoaringBitmap {
        let mut bitmap = RoaringBitmap::new();
        let mut insert = |digests: &[pb::FragmentDigest]| {
            bitmap.extend(digests.iter().map(|digest| digest.id as u32));
        };
        for stored in &self.0.stored {
            match &stored.record {
                Record::LegacyVersion(version) => {
                    for group in &version.groups {
                        insert(&group.old_fragments);
                        insert(&group.new_fragments);
                    }
                }
                Record::Transition(transition) => {
                    insert(&transition.sources);
                    insert(&transition.destinations);
                }
                Record::Uninterpretable { .. } => {}
            }
        }
        bitmap
    }

    fn lift(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        Ok(self)
    }

    fn append(&self, rewrite: &RewriteRecord) -> Result<Arc<dyn FriImpl>> {
        self.0
            .require_fully_interpretable("append transitions to")?;
        // Rewriting such a source materializes the overlaid values, while an
        // index built before the overlay keeps the source as provenance and
        // would serve the destination through translation with the old
        // values. The legacy format handles this by dropping destination ids
        // from stale bitmaps, which has no effect on tagged provenance.
        if let Some(overlaid) = rewrite.overlaid_sources().first() {
            return Err(Error::not_supported(format!(
                "source fragment {overlaid} carries data overlay files; a rewrite recording a \
                 tagged fragment reuse transition would materialize the overlaid values while \
                 indices keep translated coverage over the old addresses. Compact the overlays \
                 away or rebuild the covering indices eagerly first"
            )));
        }
        // A rewrite merged onto a history that moved on since its intent was
        // built could only find itself recorded if its sources were consumed
        // twice, which the lineage check also refuses; this names the cause.
        let ids = |digests: &[pb::FragmentDigest]| -> Vec<u64> {
            digests.iter().map(|digest| digest.id).collect()
        };
        for transition in rewrite.transitions() {
            let is_recorded = self.0.stored.iter().any(|stored| {
                matches!(&stored.record, Record::Transition(existing)
                    if ids(&existing.sources) == ids(&transition.sources)
                        && ids(&existing.destinations) == ids(&transition.destinations))
            });
            if is_recorded {
                return Err(Error::invalid_input(format!(
                    "the fragment reuse history already records a transition from fragments \
                     {:?}; a rewrite cannot append it again",
                    ids(&transition.sources)
                )));
            }
        }
        let mut content = BytesMut::from(concat(&self.0.stored));
        for transition in rewrite.transitions() {
            content.extend_from_slice(&length_delimited(
                TRANSITION_FIELD,
                &transition.encode_to_vec(),
            ));
        }
        self.rebuilt(content.freeze()).map_err(invalid_edit)
    }

    fn remap_bases(
        self: Arc<Self>,
        remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Arc<dyn FriImpl>> {
        self.0.require_fully_interpretable("relocate")?;
        let mut records = Vec::with_capacity(self.0.stored.len());
        for stored in &self.0.stored {
            let raw = match &stored.record {
                Record::Transition(pb::Transition {
                    mapping: Some(transition::Mapping::StablePartition(_)),
                    ..
                }) => {
                    let mut field = stored.raw.clone();
                    let (tag, payload) = next_field(&mut field)?;
                    let patched =
                        patch_transition_stable_partition(&require_message(tag, payload)?, remap)?;
                    length_delimited(TRANSITION_FIELD, &patched).into()
                }
                _ => stored.raw.clone(),
            };
            records.push(StoredRecord {
                raw,
                record: stored.record.clone(),
            });
        }
        self.rebuilt(concat(&records))
    }

    fn on_stable_row_ids(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        Err(Error::not_supported(
            "the table requires a tagged fragment reuse history, which cannot describe a table \
             with stable row IDs: such a table supports only index_version 0",
        ))
    }

    /// The legacy rule, per transition: a segment is caught up with a
    /// transition it covers no fragment of, or one it was built after
    /// without covering any of its sources.
    ///
    /// Covering only destinations does not prove the segment was built after
    /// the rewrite: a bitmap a legacy reader projected onto the destinations
    /// still describes files holding source addresses. Its stamp does: such
    /// a segment was built before the rewrite committed, and a transition is
    /// stamped no earlier than the version the rewrite committed on top of
    /// (its legacy version once lifted, otherwise the entry's version, which
    /// every later write of the entry only raises). The comparison is strict:
    /// a segment stamped at that version still read the sources.
    ///
    /// A dropped unknown transition leaves no record of which fragments it
    /// touched, so no segment can be proven caught up with it.
    fn is_caught_up(&self, entry_version: u64, segment: &SegmentSummary) -> bool {
        let ledger = &self.0.ledger;
        let Some(bitmap) = &segment.fragment_bitmap else {
            return false;
        };
        let covers = |digests: &[pb::FragmentDigest]| {
            digests
                .iter()
                .any(|digest| bitmap.contains(digest.id as u32))
        };
        !ledger.has_unsupported_transitions()
            && ledger.transitions().iter().all(|transition| {
                let covers_sources = covers(transition.sources());
                if !covers_sources && !covers(transition.destinations()) {
                    return true;
                }
                let recorded_at = transition.legacy_dataset_version().unwrap_or(entry_version);
                !covers_sources && segment.dataset_version > recorded_at
            })
    }

    fn trim(self: Arc<Self>, live: &[Vec<SegmentSummary>]) -> Result<Arc<dyn FriImpl>> {
        self.0.require_fully_interpretable("trim")?;
        let retained = trim::retention(&self.0.stored, live);
        if retained.iter().all(|kept| *kept) {
            return Ok(self);
        }
        let kept: Vec<StoredRecord> = self
            .0
            .stored
            .iter()
            .zip(retained)
            .filter(|(_, kept)| *kept)
            .map(|(stored, _)| stored.clone())
            .collect();
        self.rebuilt(concat(&kept))
    }

    fn needs_translation_support(&self, context: &ReadContext, segment: &SegmentSummary) -> bool {
        !self.is_direct(context, segment)
    }

    fn plan(self: Arc<Self>, context: &ReadContext, group: &[SegmentSummary]) -> Vec<ReadPlan> {
        let ledger = &self.0.ledger;
        let history: Arc<dyn FriImpl> = self.clone();
        // A fragment of the snapshot that a transition still consumes ends a
        // translated path instead of being followed through the rewrite.
        let stops: RoaringBitmap = ledger
            .transitions()
            .iter()
            .flat_map(|transition| transition.sources())
            .map(|source| source.id as u32)
            .filter(|&fragment| context.live_fragments.contains(fragment))
            .collect();
        let mut plans = vec![ReadPlan::Excluded; group.len()];
        // Only segments the reader can use establish coverage: a destination
        // needs all of its contributing sources.
        let mut members = Vec::with_capacity(group.len());
        let mut provenance = Vec::with_capacity(group.len());
        for (position, segment) in group.iter().enumerate() {
            let Some(bitmap) = &segment.fragment_bitmap else {
                continue;
            };
            if ledger.has_unsupported_transitions()
                || (!self.is_direct(context, segment)
                    && segment.translation != TranslationSupport::Supported)
            {
                continue;
            }
            members.push(position);
            provenance.push(bitmap.clone());
        }
        let parts = segment_plans(ledger, context.live_fragments, &provenance);
        for ((position, bitmap), parts) in members.into_iter().zip(&provenance).zip(parts) {
            plans[position] = if bitmap.is_empty() {
                // Nothing needs translating, but the segment's files may still
                // hold the rows whose coverage was withdrawn: never direct.
                ReadPlan::Translate(TranslationPlan::nothing(history.clone()))
            } else if self.is_direct(context, &group[position]) {
                ReadPlan::Direct
            } else if parts.coverage.is_empty() {
                ReadPlan::Translate(TranslationPlan::nothing(history.clone()))
            } else {
                ReadPlan::Translate(TranslationPlan::new(
                    history.clone(),
                    parts.coverage,
                    parts.excluded,
                    stops.clone(),
                    &parts.path,
                ))
            };
        }
        plans
    }

    fn compose(
        self: Arc<Self>,
        context: &RewriteContext,
        segment: &SegmentSummary,
    ) -> Result<Composition> {
        compose::compose(self, context, segment)
    }

    fn bind(&self, plan: TranslationPlan, mappings: Arc<dyn MappingSource>) -> Remapper {
        if let Some(path) = plan.rewrite_path() {
            return Remapper::from_backing(Arc::new(compose::RewriteRemapper::new(
                self.0.ledger.clone(),
                path.clone(),
                plan.coverage().clone(),
                mappings,
            )));
        }
        Remapper::from_backing(Arc::new(TaggedRemapper {
            ledger: self.0.ledger.clone(),
            coverage: plan.coverage().clone(),
            excluded: plan.excluded_fragments().clone(),
            stops: plan.stops().clone(),
            mappings,
        }))
    }
}

/// Translation addresses are handed to transitions in batches of at most
/// this many, bounding the buffers one batch holds.
const TRANSLATE_BATCH: usize = 64 * 1024;

/// One segment's translation under a tagged history: each address follows
/// the transitions that consume its fragment until it reaches a fragment of
/// the snapshot, and is kept only when it lands in the segment's coverage.
#[derive(Debug)]
struct TaggedRemapper {
    ledger: Arc<FragReuseLedger>,
    coverage: RoaringBitmap,
    /// Fragments sibling segments cover directly: a path entering one
    /// belongs to the sibling, so the address is dropped.
    excluded: RoaringBitmap,
    /// Live fragments a transition still consumes: a path reaching one
    /// ends there instead of following the rewrite, so the address keeps
    /// pointing into it.
    stops: RoaringBitmap,
    mappings: Arc<dyn MappingSource>,
}

impl TaggedRemapper {
    /// The transition an address in `fragment` moves through next, or `None`
    /// when its path ends there.
    fn next_hop(&self, fragment: u32) -> Option<usize> {
        if self.stops.contains(fragment) {
            return None;
        }
        self.ledger.consumer(fragment)
    }

    async fn hop(&self, transition: usize, rows: &[u64]) -> Result<Vec<Option<u64>>> {
        let transition = &self.ledger.transitions()[transition];
        match transition.mapping() {
            Mapping::OrderedCompaction(remap) => rows
                .iter()
                .map(|&row| {
                    remap.get(row).ok_or_else(|| {
                        Error::invalid_input(format!("row ID {row} is outside compaction mapping"))
                    })
                })
                .collect(),
            Mapping::StablePartition(partition) => {
                translate_stable_partition(
                    self.mappings.as_ref(),
                    &MappingRef::of(partition),
                    transition.sources(),
                    transition.destinations(),
                    rows,
                )
                .await
            }
        }
    }

    /// Translate one batch. Addresses waiting on the same transition are
    /// translated together, so a lazily read mapping is read once per batch.
    async fn translate_batch(&self, addrs: &mut [Option<u64>]) -> Result<()> {
        let mut pending: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (position, slot) in addrs.iter_mut().enumerate() {
            let Some(addr) = *slot else {
                continue;
            };
            let fragment = RowAddress::from(addr).fragment_id();
            if self.coverage.contains(fragment) {
                continue;
            }
            match self.next_hop(fragment) {
                Some(transition) => pending.entry(transition).or_default().push(position),
                None => *slot = None,
            }
        }
        while let Some((transition, positions)) = pending.pop_first() {
            let rows = positions
                .iter()
                .map(|&position| {
                    addrs[position].ok_or_else(|| {
                        Error::internal("deleted address queued for FRI translation")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let translated = self.hop(transition, &rows).await?;
            for (position, addr) in positions.into_iter().zip(translated) {
                let addr = addr.filter(|addr| {
                    !self
                        .excluded
                        .contains(RowAddress::from(*addr).fragment_id())
                });
                addrs[position] = addr;
                if let Some(addr) = addr
                    && let Some(next) = self.next_hop(RowAddress::from(addr).fragment_id())
                {
                    pending.entry(next).or_default().push(position);
                }
            }
        }
        for slot in addrs.iter_mut() {
            if slot
                .is_some_and(|addr| !self.coverage.contains(RowAddress::from(addr).fragment_id()))
            {
                *slot = None;
            }
        }
        Ok(())
    }
}

impl RemapperBacking for TaggedRemapper {
    fn fragment_needs_translation(&self, fragment: u32) -> bool {
        !self.coverage.contains(fragment)
    }

    fn translate_fragments(&self, fragments: &RoaringBitmap) -> RoaringBitmap {
        let mut translated = fragments & &self.coverage;
        let mut pending: Vec<u32> = (fragments - &self.coverage).iter().collect();
        let mut visited = RoaringBitmap::new();
        while let Some(fragment) = pending.pop() {
            if !visited.insert(fragment) {
                continue;
            }
            let Some(transition) = self.next_hop(fragment) else {
                continue;
            };
            for destination in self.ledger.transitions()[transition].destinations() {
                let destination = destination.id as u32;
                if self.excluded.contains(destination) {
                    continue;
                }
                if self.next_hop(destination).is_some() {
                    pending.push(destination);
                } else if self.coverage.contains(destination) {
                    translated.insert(destination);
                }
            }
        }
        translated
    }

    fn translate_in_place<'a>(&'a self, addrs: &'a mut [Option<u64>]) -> TranslateFuture<'a> {
        Box::pin(async move {
            for batch in addrs.chunks_mut(TRANSLATE_BATCH) {
                self.translate_batch(batch).await?;
            }
            Ok(())
        })
    }
}

/// Rules the lineage decode does not see. Each stable-partition transition
/// owns its row map: cleanup deletes a map once no retained transition
/// references it, so a shared map would be deleted while still in use.
fn validate_records(records: &[StoredRecord]) -> Result<()> {
    let mut map_ids = HashSet::new();
    for stored in records {
        if let Record::Transition(pb::Transition {
            mapping: Some(transition::Mapping::StablePartition(partition)),
            ..
        }) = &stored.record
            && !map_ids.insert(partition.map_id.as_str())
        {
            return Err(corrupt(format!(
                "stable-partition row map {} is referenced by more than one transition",
                partition.map_id
            )));
        }
    }
    Ok(())
}

/// Patch a transition's stable-partition mapping, copying every other
/// field's bytes as written.
fn patch_transition_stable_partition(
    raw: &Bytes,
    remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
) -> Result<Vec<u8>> {
    let mut remaining = raw.clone();
    let mut output = Vec::with_capacity(raw.len() + 8);
    while remaining.has_remaining() {
        let start = raw.len() - remaining.len();
        let (tag, payload) = next_field(&mut remaining)?;
        if tag == STABLE_PARTITION_FIELD {
            let patched = patch_stable_partition_base(&require_message(tag, payload)?, remap)?;
            output.extend_from_slice(&length_delimited(STABLE_PARTITION_FIELD, &patched));
        } else {
            output.extend_from_slice(&raw[start..raw.len() - remaining.len()]);
        }
    }
    Ok(output)
}

/// Drop `base_id` wherever it occurs (the last occurrence is the effective
/// value) and append the remapped value; every other byte is kept.
fn patch_stable_partition_base(
    raw: &Bytes,
    remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
) -> Result<Vec<u8>> {
    let mut remaining = raw.clone();
    let mut output = Vec::with_capacity(raw.len() + 8);
    let mut existing = None;
    while remaining.has_remaining() {
        let start = raw.len() - remaining.len();
        let mut peek = remaining.clone();
        let (tag, wire_type) = decode_key(&mut peek).map_err(|e| corrupt(e.to_string()))?;
        if tag == STABLE_PARTITION_BASE_ID_FIELD {
            if wire_type != WireType::Varint {
                return Err(corrupt("stable partition base_id must be a varint"));
            }
            let value = decode_varint(&mut peek).map_err(|e| corrupt(e.to_string()))?;
            existing =
                Some(u32::try_from(value).map_err(|_| {
                    corrupt(format!("stable partition base_id {value} overflows u32"))
                })?);
            remaining = peek;
        } else {
            next_field(&mut remaining)?;
            output.extend_from_slice(&raw[start..raw.len() - remaining.len()]);
        }
    }
    if let Some(base_id) = remap(existing)? {
        prost::encoding::encode_key(
            STABLE_PARTITION_BASE_ID_FIELD,
            WireType::Varint,
            &mut output,
        );
        prost::encoding::encode_varint(u64::from(base_id), &mut output);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_index::frag_reuse::history::tests::{
        MAP_A, MAP_B, digest, ordered, partition, whole,
    };
    use crate::system_index::frag_reuse::history::{FragReuseHistory, RawHistory};
    use crate::system_index::frag_reuse::ledger::Mapping;
    use crate::system_index::frag_reuse::plan::RowDomain;
    use lance_core::utils::stable_partition::CountsMatrix;

    #[test]
    fn appended_transitions_are_validated_against_the_whole_lineage() {
        let history = FragReuseHistory::empty(HistoryFormat::Tagged)
            .append(RewriteRecord::new(
                1,
                vec![ordered(&[whole(0, 2)], &[(1, 2)])],
            ))
            .unwrap();
        let error = history
            .append(RewriteRecord::new(
                1,
                vec![ordered(&[whole(0, 2)], &[(2, 2)])],
            ))
            .unwrap_err();
        assert!(error.to_string().contains("duplicate consumer"), "{error}");
    }

    /// Ordered compactions read no row map.
    #[derive(Debug)]
    struct NoRowMaps;

    #[async_trait::async_trait]
    impl MappingSource for NoRowMaps {
        async fn counts(&self, mapping: &MappingRef) -> Result<Arc<CountsMatrix>> {
            Err(Error::internal(format!("unexpected row map {mapping:?}")))
        }

        async fn read(
            &self,
            mapping: &MappingRef,
            _rows: std::ops::Range<u64>,
        ) -> Result<arrow_array::UInt16Array> {
            Err(Error::internal(format!("unexpected row map {mapping:?}")))
        }
    }

    /// A path that enters a fragment a sibling segment covers belongs to the
    /// sibling: the address is dropped there, even when the path continues
    /// into this segment's coverage. An address already in an excluded
    /// fragment is still translated, since a projected bitmap may name it.
    #[tokio::test]
    async fn paths_into_a_sibling_fragment_are_dropped() {
        // 0 -> 1 -> 2 -> 3, with fragment 9 never rewritten.
        let mut history = FragReuseHistory::empty(HistoryFormat::Tagged);
        for (version, (source, destination)) in [(0, 1), (1, 2), (2, 3)].into_iter().enumerate() {
            history = history
                .append(RewriteRecord::new(
                    version as u64 + 1,
                    vec![ordered(&[whole(source, 2)], &[(destination, 2)])],
                ))
                .unwrap();
        }
        let remapper = Remapper::from_backing(Arc::new(TaggedRemapper {
            ledger: history.ledger().clone(),
            coverage: RoaringBitmap::from_iter([3, 9]),
            excluded: RoaringBitmap::from_iter([2]),
            stops: RoaringBitmap::new(),
            mappings: Arc::new(NoRowMaps),
        }));
        let address = |fragment| u64::from(RowAddress::new_from_parts(fragment, 1));

        let mut addrs = [0, 1, 2, 3, 9].map(|fragment| Some(address(fragment)));
        remapper.translate_in_place(&mut addrs).await.unwrap();
        assert_eq!(
            addrs,
            [
                None,
                None,
                Some(address(3)),
                Some(address(3)),
                Some(address(9))
            ]
        );
        assert_eq!(
            remapper.translate_fragments(&RoaringBitmap::from_iter([0, 1])),
            RoaringBitmap::new()
        );
        assert_eq!(
            remapper.translate_fragments(&RoaringBitmap::from_iter([2, 9])),
            RoaringBitmap::from_iter([3, 9])
        );
        assert!(remapper.fragment_needs_translation(2));
        assert!(!remapper.fragment_needs_translation(3));
    }

    /// A path that reaches a live fragment a transition still consumes ends
    /// there: the address keeps pointing into it rather than following the
    /// rewrite on.
    #[tokio::test]
    async fn paths_end_at_a_stop_fragment() {
        // 0 -> 1 -> 2 -> 3, with fragment 1 still live.
        let mut history = FragReuseHistory::empty(HistoryFormat::Tagged);
        for (version, (source, destination)) in [(0, 1), (1, 2), (2, 3)].into_iter().enumerate() {
            history = history
                .append(RewriteRecord::new(
                    version as u64 + 1,
                    vec![ordered(&[whole(source, 2)], &[(destination, 2)])],
                ))
                .unwrap();
        }
        let remapper = Remapper::from_backing(Arc::new(TaggedRemapper {
            ledger: history.ledger().clone(),
            coverage: RoaringBitmap::from_iter([1, 3]),
            excluded: RoaringBitmap::new(),
            stops: RoaringBitmap::from_iter([1]),
            mappings: Arc::new(NoRowMaps),
        }));
        let address = |fragment| u64::from(RowAddress::new_from_parts(fragment, 1));

        let mut addrs = [0, 1, 2].map(|fragment| Some(address(fragment)));
        remapper.translate_in_place(&mut addrs).await.unwrap();
        assert_eq!(
            addrs,
            [Some(address(1)), Some(address(1)), Some(address(3))]
        );
        assert_eq!(
            remapper.translate_fragments(&RoaringBitmap::from_iter([0])),
            RoaringBitmap::from_iter([1])
        );
        assert_eq!(
            remapper.translate_fragments(&RoaringBitmap::from_iter([1])),
            RoaringBitmap::from_iter([1])
        );
        assert_eq!(
            remapper.translate_fragments(&RoaringBitmap::from_iter([2])),
            RoaringBitmap::from_iter([3])
        );
    }

    /// A tagged transition records no version on the wire, so the caught-up
    /// rule bounds it by the entry's `dataset_version`, which every later
    /// rewrite of the entry raises. Under continuous compaction a segment
    /// built after a transition onto its destinations therefore stops being
    /// read directly once another rewrite lands, and is translated instead
    /// (correct, at the cost of translating). Trimming does not depend on
    /// that bound: it releases the transition once no logical index derives
    /// coverage from its sources, so the history still drains.
    #[test]
    fn the_entry_version_bounds_reads_but_not_the_drain() {
        use crate::system_index::frag_reuse::plan::{RowDomain, TranslationSupport};
        let first = FragReuseHistory::empty(HistoryFormat::Tagged)
            .append(RewriteRecord::new(
                5,
                vec![ordered(&[whole(0, 2)], &[(1, 2)])],
            ))
            .unwrap();
        // Rebuilt over the destination at version 6, after the rewrite.
        let segment = SegmentSummary {
            uuid: uuid::Uuid::new_v4(),
            fragment_bitmap: Some(RoaringBitmap::from_iter([1])),
            dataset_version: 6,
            domain: RowDomain::Address,
            translation: TranslationSupport::Supported,
        };
        assert!(first.is_caught_up(5, &segment));
        // An unrelated rewrite re-stamps the entry at version 9.
        let second = first
            .append(RewriteRecord::new(
                9,
                vec![ordered(&[whole(7, 2)], &[(8, 2)])],
            ))
            .unwrap();
        assert!(!second.is_caught_up(9, &segment));
        let live = RoaringBitmap::from_iter([1, 8]);
        let context = ReadContext {
            entry_version: 9,
            live_fragments: &live,
        };
        let plan = second
            .plan(&context, std::slice::from_ref(&segment))
            .remove(0);
        let ReadPlan::Translate(plan) = plan else {
            panic!("the upper bound translates the segment, got {plan:?}");
        };
        assert_eq!(plan.coverage(), &RoaringBitmap::from_iter([1]));
        // Nothing derives coverage from either transition's sources.
        let trimmed = second.trim(&[vec![segment]]).unwrap();
        assert_eq!(trimmed.records().count(), 0);
    }

    #[test]
    fn a_row_map_belongs_to_one_transition() {
        let history = FragReuseHistory::empty(HistoryFormat::Tagged)
            .append(RewriteRecord::new(1, vec![partition(0, 1, MAP_A)]))
            .unwrap();
        let error = history
            .append(RewriteRecord::new(1, vec![partition(2, 3, MAP_A)]))
            .unwrap_err();
        assert!(
            error.to_string().contains("more than one transition"),
            "{error}"
        );
        history
            .append(RewriteRecord::new(1, vec![partition(2, 3, MAP_B)]))
            .unwrap();

        let content: Vec<u8> = [partition(0, 1, MAP_A), partition(2, 3, MAP_A)]
            .iter()
            .flat_map(|transition| length_delimited(TRANSITION_FIELD, &transition.encode_to_vec()))
            .collect();
        let error = RawHistory::new(HistoryFormat::Tagged, content.into())
            .validate()
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(
            error.to_string().contains("more than one transition"),
            "{error}"
        );
    }

    /// A lifted legacy transition is caught up by its own stamp; a native one
    /// has none, so it is held to the entry's version.
    #[test]
    fn caught_up_compares_against_each_transitions_version() {
        const ENTRY_VERSION: u64 = 10;
        let history = FragReuseHistory::empty(HistoryFormat::Legacy)
            .append(RewriteRecord::new(
                5,
                vec![ordered(&[whole(0, 1)], &[(1, 1)])],
            ))
            .unwrap()
            .lift()
            .unwrap()
            .append(RewriteRecord::new(
                8,
                vec![ordered(&[whole(2, 1)], &[(3, 1)])],
            ))
            .unwrap();
        let live = RoaringBitmap::from_iter([1, 3]);
        let context = ReadContext {
            entry_version: ENTRY_VERSION,
            live_fragments: &live,
        };
        for (covered, dataset_version, is_caught_up) in [
            (1, 7, true),
            (3, 9, false),
            (3, ENTRY_VERSION, false),
            (3, ENTRY_VERSION + 1, true),
        ] {
            let segment = SegmentSummary {
                uuid: uuid::Uuid::new_v4(),
                fragment_bitmap: Some(RoaringBitmap::from_iter([covered])),
                dataset_version,
                domain: RowDomain::Address,
                translation: TranslationSupport::Supported,
            };
            assert_eq!(
                history.is_caught_up(ENTRY_VERSION, &segment),
                is_caught_up,
                "fragment {covered} at version {dataset_version}"
            );
            let plan = history.plan(&context, std::slice::from_ref(&segment));
            assert_eq!(
                plan[0] == ReadPlan::Direct,
                is_caught_up,
                "fragment {covered} at version {dataset_version}: {plan:?}"
            );
        }
    }

    #[test]
    fn base_relocation_keeps_unknown_nested_fields() {
        let mut partition_payload = pb::StablePartition {
            map_id: MAP_A.to_string(),
            map_size_bytes: 64,
            base_id: Some(2),
        }
        .encode_to_vec();
        partition_payload.extend_from_slice(&length_delimited(9, b"future"));
        let mut transition = pb::Transition {
            sources: vec![digest(0, 4, 0)],
            destinations: vec![digest(1, 4, 0)],
            mapping: None,
        }
        .encode_to_vec();
        transition.extend_from_slice(&length_delimited(
            STABLE_PARTITION_FIELD,
            &partition_payload,
        ));
        let history = RawHistory::new(
            HistoryFormat::Tagged,
            length_delimited(TRANSITION_FIELD, &transition).into(),
        )
        .validate()
        .unwrap();

        let relocated = history
            .remap_bases(&|base| {
                assert_eq!(base, Some(2));
                Ok(Some(7))
            })
            .unwrap();
        let Mapping::StablePartition(partition) = relocated.ledger().transitions()[0].mapping()
        else {
            panic!("not a stable partition");
        };
        assert_eq!(partition.base_id, Some(7));
        let content = relocated.content();
        let unknown = length_delimited(9, b"future");
        assert!(
            content
                .windows(unknown.len())
                .any(|window| window == unknown),
            "the unknown nested field was dropped"
        );
    }
}
