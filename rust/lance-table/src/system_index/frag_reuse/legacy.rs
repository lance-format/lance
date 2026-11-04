// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The legacy (`index_version` 0) fragment reuse history: compaction
//! versions only, ordered by dataset version, which every released reader
//! and writer understands.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use lance_core::{Error, Result};
use prost::Message;
use roaring::RoaringBitmap;

use super::FragReuseVersion;
use super::history::{
    FragReuseHistory, HistoryFormat, LEGACY_VERSION_FIELD, Record, RewriteRecord, length_delimited,
    split_records,
};
use super::imp::{FriImpl, Records, StoredRecord, concat, invalid_edit};
use super::legacy_srid::LegacySridHistory;
use super::lift;
use super::plan::{ReadContext, ReadPlan, SegmentSummary, TranslationPlan};
use crate::format::IndexMetadata;
use crate::format::pb::fragment_reuse_index_details as pb;
use crate::system_index::is_system_index;

#[derive(Debug, Default)]
pub struct LegacyHistory(Records);

impl LegacyHistory {
    pub fn decode(content: Bytes) -> Result<Self> {
        Records::decode(HistoryFormat::Legacy, content).map(Self)
    }

    /// The history a committed entry stores, repairing one an older writer
    /// left invalid (see [`repair_versions`]).
    pub fn decode_repairing(content: Bytes) -> Result<(FragReuseHistory, Vec<u64>)> {
        match Self::decode(content.clone()) {
            Ok(history) => Ok((FragReuseHistory::from_impl(Arc::new(history)), Vec::new())),
            Err(Error::CorruptFile { .. }) => {
                let versions = split_records(content)?
                    .into_iter()
                    .filter_map(|stored| match stored.record {
                        Record::LegacyVersion(version) => Some(version),
                        _ => None,
                    })
                    .collect();
                repair_versions(versions)
            }
            Err(error) => Err(error),
        }
    }
}

/// A version 0 history holding `versions`, ordered by dataset version.
pub fn from_versions(mut versions: Vec<pb::Version>) -> Result<FragReuseHistory> {
    versions.sort_by_key(|version| version.dataset_version);
    let mut content = BytesMut::new();
    for version in &versions {
        content.extend_from_slice(&length_delimited(
            LEGACY_VERSION_FIELD,
            &version.encode_to_vec(),
        ));
    }
    Ok(FragReuseHistory::from_impl(Arc::new(
        LegacyHistory::decode(content.freeze())?,
    )))
}

/// See [`FragReuseHistory::repair_legacy_versions`].
pub fn repair_versions(mut versions: Vec<pb::Version>) -> Result<(FragReuseHistory, Vec<u64>)> {
    versions.sort_by_key(|version| version.dataset_version);
    let mut kept = Vec::with_capacity(versions.len());
    let mut dropped = Vec::new();
    let mut produced: HashMap<u64, u64> = HashMap::new();
    let mut consumed: HashSet<u64> = HashSet::new();
    for version in versions {
        let fits_lineage = version.groups.iter().all(|group| {
            group.old_fragments.iter().all(|source| {
                !consumed.contains(&source.id)
                    && produced
                        .get(&source.id)
                        .is_none_or(|&rows| rows == source.physical_rows)
            }) && group.new_fragments.iter().all(|destination| {
                !produced.contains_key(&destination.id) && !consumed.contains(&destination.id)
            })
        });
        if !fits_lineage || from_versions(vec![version.clone()]).is_err() {
            dropped.push(version.dataset_version);
            continue;
        }
        for group in &version.groups {
            consumed.extend(group.old_fragments.iter().map(|source| source.id));
            produced.extend(
                group
                    .new_fragments
                    .iter()
                    .map(|destination| (destination.id, destination.physical_rows)),
            );
        }
        kept.push(version);
    }
    Ok((from_versions(kept)?, dropped))
}

/// Every fragment the reuse chain touches (old + new) across all versions. An
/// index disjoint from this set holds no row address the FRI remaps, so trimming
/// can never strand it (fragment ids are never reused).
pub fn reuse_chain_frag_bitmap(versions: &[FragReuseVersion]) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    for version in versions {
        bitmap.extend(version.old_frag_ids().iter().map(|&id| id as u32));
        bitmap.extend(version.new_frag_ids().iter().map(|&id| id as u32));
    }
    bitmap
}

/// Whether `index_meta` is caught up with one compaction version, so the
/// version can be trimmed for it. `chain_frag_bitmap` is
/// [`reuse_chain_frag_bitmap`] of the whole history. `InvalidInput` when the
/// index covers part of a rewrite group, which only a corrupted version
/// records.
pub fn is_index_remap_caught_up(
    frag_reuse_version: &FragReuseVersion,
    index_meta: &IndexMetadata,
    chain_frag_bitmap: &RoaringBitmap,
) -> Result<bool> {
    if is_system_index(index_meta) {
        return Ok(true);
    }
    if index_meta.fragment_bitmap.is_none()
        && index_meta.dataset_version > frag_reuse_version.dataset_version
    {
        log::warn!(
            "Index {} ({}) missing fragment bitmap, cannot determine if it is caught up with the fragment reuse version, consider retraining the index",
            index_meta.name,
            index_meta.uuid
        );
    }
    is_caught_up_with_version(
        frag_reuse_version.dataset_version,
        frag_reuse_version
            .groups
            .iter()
            .map(|group| group.old_frags.iter().map(|frag| frag.id).collect()),
        index_meta.fragment_bitmap.as_ref(),
        index_meta.dataset_version,
        chain_frag_bitmap,
    )
}

/// The legacy caught-up rule for one compaction version stamped
/// `version_stamp`, whose groups consume `group_sources`.
fn is_caught_up_with_version(
    version_stamp: u64,
    group_sources: impl IntoIterator<Item = Vec<u64>>,
    fragment_bitmap: Option<&RoaringBitmap>,
    segment_version: u64,
    chain_frag_bitmap: &RoaringBitmap,
) -> Result<bool> {
    // Disjoint coverage => caught up regardless of dataset_version, bypassing the
    // stale-version gate below. The chain includes NEW fragments
    // deliberately: a deferred-remap commit advances a covering index's bitmap
    // onto them before its data is remapped, so an old-frag-only check would
    // clear a still-stale index and trim a version it needs.
    if let Some(index_frag_bitmap) = fragment_bitmap
        && index_frag_bitmap.is_disjoint(chain_frag_bitmap)
    {
        return Ok(true);
    }

    // Inclusive: an index built at dataset_version predates the rewrite.
    if segment_version <= version_stamp {
        return Ok(false);
    }

    // Without a bitmap released writers took the index as caught up.
    let Some(index_frag_bitmap) = fragment_bitmap else {
        return Ok(true);
    };
    for old_frags in group_sources {
        let old_frag_in_index = old_frags
            .iter()
            .filter(|&&id| index_frag_bitmap.contains(id as u32))
            .count();
        if old_frag_in_index > 0 {
            if old_frag_in_index != old_frags.len() {
                // This should never happen because we always commit a full rewrite group
                // and we always reindex either the entire group or nothing.
                // We use invalid input to be consistent with
                // dataset::transaction::recalculate_fragment_bitmap
                return Err(Error::invalid_input(format!(
                    "The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {old_frags:?}"
                )));
            }
            return Ok(false);
        }
    }
    Ok(true)
}

impl LegacyHistory {
    fn versions(&self) -> impl Iterator<Item = &pb::Version> {
        self.0
            .stored
            .iter()
            .filter_map(|stored| match &stored.record {
                Record::LegacyVersion(version) => Some(version),
                _ => None,
            })
    }

    fn plan_segment(&self, context: &ReadContext, segment: &SegmentSummary) -> ReadPlan {
        let coverage = segment
            .fragment_bitmap
            .as_ref()
            .map(|bitmap| bitmap & context.live_fragments)
            .unwrap_or_default();
        let is_direct = segment
            .fragment_bitmap
            .as_ref()
            .is_some_and(|bitmap| !bitmap.is_empty())
            && self.is_caught_up(context.entry_version, segment);
        if is_direct {
            ReadPlan::Direct
        } else {
            ReadPlan::Translate(TranslationPlan::new(
                &self.0.ledger,
                coverage,
                RoaringBitmap::new(),
                &[],
            ))
        }
    }
}

impl FriImpl for LegacyHistory {
    fn format(&self) -> HistoryFormat {
        HistoryFormat::Legacy
    }

    fn records(&self) -> &Records {
        &self.0
    }

    fn rebuilt(&self, content: Bytes) -> Result<Arc<dyn FriImpl>> {
        Ok(Arc::new(Self::decode(content)?))
    }

    /// Released writers stored the destinations of every compaction version.
    fn fragment_bitmap(&self) -> RoaringBitmap {
        let mut bitmap = RoaringBitmap::new();
        for stored in &self.0.stored {
            if let Record::LegacyVersion(version) = &stored.record {
                for group in &version.groups {
                    bitmap.extend(group.new_fragments.iter().map(|digest| digest.id as u32));
                }
            }
        }
        bitmap
    }

    fn lift(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        lift::lift_legacy(&self.0)
    }

    /// Insert the rewrite's compaction version, keeping versions ordered by
    /// `dataset_version` as version 0 readers expect.
    fn append(&self, rewrite: &RewriteRecord) -> Result<Arc<dyn FriImpl>> {
        let version = lift::legacy_version(rewrite)?;
        let position = self
            .0
            .stored
            .iter()
            .position(|stored| {
                matches!(&stored.record, Record::LegacyVersion(existing)
                    if existing.dataset_version > version.dataset_version)
            })
            .unwrap_or(self.0.stored.len());
        let mut records = self.0.stored.clone();
        records.insert(
            position,
            StoredRecord {
                raw: length_delimited(LEGACY_VERSION_FIELD, &version.encode_to_vec()).into(),
                record: Record::LegacyVersion(version),
            },
        );
        self.rebuilt(concat(&records)).map_err(invalid_edit)
    }

    // A version 0 history has no stable partitions, so there is nothing to
    // restamp once it is known to hold no record a newer writer added.
    fn remap_bases(
        self: Arc<Self>,
        _remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Arc<dyn FriImpl>> {
        self.0.require_fully_interpretable("relocate")?;
        Ok(self)
    }

    fn on_stable_row_ids(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        Ok(Arc::new(LegacySridHistory::new(self)))
    }

    /// Caught up with every compaction version under the rule trimming
    /// applies; a version the segment covers part of a group of is never
    /// caught up with.
    fn is_caught_up(&self, _entry_version: u64, segment: &SegmentSummary) -> bool {
        let mut chain = RoaringBitmap::new();
        for version in self.versions() {
            for group in &version.groups {
                chain.extend(group.old_fragments.iter().map(|digest| digest.id as u32));
                chain.extend(group.new_fragments.iter().map(|digest| digest.id as u32));
            }
        }
        self.versions().all(|version| {
            matches!(
                is_caught_up_with_version(
                    version.dataset_version,
                    version.groups.iter().map(|group| group
                        .old_fragments
                        .iter()
                        .map(|digest| digest.id)
                        .collect()),
                    segment.fragment_bitmap.as_ref(),
                    segment.dataset_version,
                    &chain,
                ),
                Ok(true)
            )
        })
    }

    // The legacy reader applies the history to every segment it opens and
    // trusts every kind to translate.
    fn needs_translation_support(&self, _context: &ReadContext, _segment: &SegmentSummary) -> bool {
        false
    }

    /// Every segment that is not caught up translates its live coverage: the
    /// stored bitmap a legacy reader projected onto destinations already
    /// names it.
    fn plan(&self, context: &ReadContext, group: &[SegmentSummary]) -> Vec<ReadPlan> {
        group
            .iter()
            .map(|segment| self.plan_segment(context, segment))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_index::frag_reuse::history::tests::{
        address, legacy, ordered, translate, whole,
    };

    #[test]
    fn legacy_versions_stay_ordered_by_dataset_version() {
        let history = FragReuseHistory::empty(HistoryFormat::Legacy)
            .append(RewriteRecord::new(
                5,
                vec![ordered(&[whole(0, 1)], &[(1, 1)])],
            ))
            .unwrap()
            .append(RewriteRecord::new(
                3,
                vec![ordered(&[whole(2, 1)], &[(3, 1)])],
            ))
            .unwrap();
        let versions: Vec<u64> = history
            .records()
            .map(|record| match record {
                Record::LegacyVersion(version) => version.dataset_version,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(versions, [3, 5]);
    }

    #[test]
    fn repair_drops_versions_that_break_the_kept_lineage() {
        let first = legacy(1, &[whole(0, 2)], &[(1, 2)]);
        // Consumes fragment 0 a second time.
        let reconsumes = legacy(2, &[whole(0, 2)], &[(2, 2)]);
        // Produces fragment 1 a second time.
        let reproduces = legacy(3, &[whole(5, 2)], &[(1, 2)]);
        // Invalid on its own: two survivors recorded into one destination row.
        let mut miscounted = legacy(4, &[whole(6, 2)], &[(7, 2)]);
        miscounted.groups[0].new_fragments[0].physical_rows = 1;
        let chained = legacy(5, &[whole(1, 2)], &[(8, 2)]);

        let error = FragReuseHistory::from_legacy_versions(vec![
            first.clone(),
            reconsumes.clone(),
            reproduces.clone(),
            miscounted.clone(),
            chained.clone(),
        ])
        .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");

        let (repaired, dropped) = FragReuseHistory::repair_legacy_versions(vec![
            chained, miscounted, reproduces, reconsumes, first,
        ])
        .unwrap();
        assert_eq!(dropped, [2, 3, 4]);
        let kept: Vec<u64> = repaired
            .records()
            .map(|record| match record {
                Record::LegacyVersion(version) => version.dataset_version,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(kept, [1, 5]);
        assert_eq!(
            translate(repaired.ledger(), address(0, 1)),
            Some(address(8, 1))
        );
    }

    fn reuse_version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
        let digest = |id| super::super::FragDigest {
            id,
            physical_rows: 100,
            num_deleted_rows: 0,
        };
        FragReuseVersion {
            dataset_version,
            groups: vec![super::super::FragReuseGroup {
                changed_row_addrs: Vec::new(),
                old_frags: old.iter().copied().map(digest).collect(),
                new_frags: new.iter().copied().map(digest).collect(),
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
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    /// The catch-up determination must not pin the FRI on an index that is
    /// simply unrelated to the compaction, while still retaining versions that a
    /// covering-but-not-yet-remapped index needs.
    #[test]
    fn caught_up_uses_fragment_coverage_not_only_version() {
        // A reuse version at dataset_version 10 rewrote fragments [4, 5] -> [6].
        let version = reuse_version(10, &[4, 5], &[6]);
        let chain = reuse_chain_frag_bitmap(std::slice::from_ref(&version));
        let caught_up = |dataset_version, covered: &[u32]| {
            is_index_remap_caught_up(&version, &index_covering(dataset_version, covered), &chain)
                .unwrap()
        };

        // Non-covering, stale version: touches none of the rewritten frags, so
        // caught up despite version 5 < 10.
        assert!(caught_up(5, &[1, 2, 3]));
        // Still holds an old fragment: not caught up.
        assert!(!caught_up(5, &[1, 4, 5]));
        // Bitmap advanced onto the new fragment but data not yet remapped: not
        // caught up (why the chain must include new frags).
        assert!(!caught_up(5, &[1, 6]));
        // Built at version 10 itself: still pre-rewrite data, not caught up.
        assert!(!caught_up(10, &[1, 6]));
        // Built or remapped after version 10: caught up.
        assert!(caught_up(11, &[1, 6]));
        // Partly covering a rewrite group only a corrupted version records.
        let error =
            is_index_remap_caught_up(&version, &index_covering(11, &[4]), &chain).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }

    /// The chain spans every reuse version, not just the one being checked: a
    /// stale index touching only a *later* version's fragment must still fall to
    /// the version gate (a per-version chain would wrongly clear it).
    #[test]
    fn caught_up_uses_whole_reuse_chain() {
        let v1 = reuse_version(10, &[4, 5], &[6]);
        let v2 = reuse_version(11, &[6], &[7]);
        let chain = reuse_chain_frag_bitmap(&[v1.clone(), v2]);
        assert!(!is_index_remap_caught_up(&v1, &index_covering(5, &[1, 7]), &chain).unwrap());
    }

    /// Whole-fragment removal (every row deleted, no replacement): an index
    /// emptied by the deletion has an empty bitmap and must count as caught up --
    /// it holds only dead rows -- else its stale version pins the removed-fragment
    /// version forever.
    #[test]
    fn caught_up_handles_fragment_removal() {
        let version = reuse_version(20, &[7], &[]);
        let chain = reuse_chain_frag_bitmap(std::slice::from_ref(&version));
        assert!(is_index_remap_caught_up(&version, &index_covering(5, &[]), &chain).unwrap());
        assert!(!is_index_remap_caught_up(&version, &index_covering(5, &[7]), &chain).unwrap());
    }

    /// Released writers did not always record a bitmap; such an index is
    /// taken as caught up once built after the version, as they took it.
    #[test]
    fn caught_up_without_a_bitmap_after_the_version() {
        let version = reuse_version(10, &[4], &[6]);
        let chain = reuse_chain_frag_bitmap(std::slice::from_ref(&version));
        let mut index = index_covering(11, &[]);
        index.fragment_bitmap = None;
        assert!(is_index_remap_caught_up(&version, &index, &chain).unwrap());
        index.dataset_version = 10;
        assert!(!is_index_remap_caught_up(&version, &index, &chain).unwrap());
    }
}
