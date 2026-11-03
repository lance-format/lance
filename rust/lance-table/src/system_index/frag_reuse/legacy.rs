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

use super::history::{
    FragReuseHistory, HistoryFormat, LEGACY_VERSION_FIELD, Record, length_delimited, split_records,
};
use super::imp::{FriImpl, Records, StoredRecord, concat, invalid_edit};
use super::tagged::TaggedHistory;
use crate::format::pb::fragment_reuse_index_details as pb;

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

/// See [`FragReuseHistory::from_legacy_versions`].
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

    // Version 0 records are valid tagged records, so the bytes are unchanged.
    fn into_tagged(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        Ok(Arc::new(TaggedHistory::decode(concat(&self.0.stored))?))
    }

    fn append_transitions(&self, _transitions: &[pb::Transition]) -> Result<Arc<dyn FriImpl>> {
        Err(Error::invalid_input(
            "transitions require the tagged fragment reuse format; convert the history with \
             `into_tagged` first",
        ))
    }

    fn insert_legacy_version(&self, version: pb::Version) -> Result<Arc<dyn FriImpl>> {
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
    fn remap_stable_partition_bases(
        self: Arc<Self>,
        _remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Arc<dyn FriImpl>> {
        self.0.require_fully_interpretable("relocate")?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_index::frag_reuse::history::tests::{address, legacy, translate, whole};

    #[test]
    fn legacy_versions_stay_ordered_by_dataset_version() {
        let history = FragReuseHistory::empty(HistoryFormat::Legacy)
            .insert_legacy_version(legacy(5, &[whole(0, 1)], &[(1, 1)]))
            .unwrap()
            .insert_legacy_version(legacy(3, &[whole(2, 1)], &[(3, 1)]))
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
}
