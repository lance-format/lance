// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Record which MemWAL SSTables have been compacted into the base table.

use super::Footprint;
use super::apply::ApplyState;
use crate::format::pb;
use crate::system_index::mem_wal::{CompactedSsTable, MEM_WAL_INDEX_NAME};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Mark MemWAL SSTables as compacted into the base table.
///
/// The rows were already readable through the WAL, so this records where they
/// are rather than changing them. A shard's generation may only move forward,
/// and the table must already carry a MemWAL index: progress against a shard
/// nothing corroborates is rejected rather than invented.
///
/// This is the one action that edits the MemWAL system index rather than the
/// data, which is why it exists at all: the index is a segment like any other,
/// but its contents are compaction bookkeeping that no
/// [`AddIndexSegment`](super::AddIndexSegment) could express as a delta.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UpdateCompactedSsTables {
    pub compacted_sstables: Vec<CompactedSsTable>,
}

impl UpdateCompactedSsTables {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        if self.compacted_sstables.is_empty() {
            return Err(Error::invalid_input(
                "UpdateCompactedSsTables names no SSTable, so there is no progress to record",
            ));
        }
        state.update_compacted_sstables(self.compacted_sstables.clone())
    }

    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// A rewrite of the whole MemWAL index. Unlike a segment joining an
    /// ordinary index, this restates the index entry outright, so a concurrent
    /// writer touching that index at all would lose what this recorded --
    /// which is what an unstated reach claims.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.rewrite_index(MEM_WAL_INDEX_NAME.to_string());
    }
}

impl From<&UpdateCompactedSsTables> for pb::UpdateCompactedSsTables {
    fn from(value: &UpdateCompactedSsTables) -> Self {
        Self {
            compacted_sstables: value
                .compacted_sstables
                .iter()
                .map(pb::CompactedSsTable::from)
                .collect(),
        }
    }
}

impl TryFrom<pb::UpdateCompactedSsTables> for UpdateCompactedSsTables {
    type Error = Error;

    fn try_from(message: pb::UpdateCompactedSsTables) -> Result<Self> {
        Ok(Self {
            compacted_sstables: message
                .compacted_sstables
                .into_iter()
                .map(CompactedSsTable::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_index::mem_wal::{
        MemWalIndexDetails, load_mem_wal_index_details, new_mem_wal_index_meta,
    };
    use crate::transaction::action::test_support::{apply_with_indices, backed_manifest};
    use crate::transaction::action::{Action, CompositeOperation, UserAction};
    use crate::transaction::test_support::sample_index_metadata;
    use uuid::Uuid;

    fn update(sstables: Vec<(u128, u64)>) -> Action {
        Action::UpdateCompactedSsTables(UpdateCompactedSsTables {
            compacted_sstables: sstables
                .into_iter()
                .map(|(shard, generation)| {
                    CompactedSsTable::new(Uuid::from_u128(shard), generation)
                })
                .collect(),
        })
    }

    /// The compaction progress the MemWAL index records, as (shard, generation)
    /// pairs sorted by shard.
    fn progress(indices: &[crate::format::IndexMetadata]) -> Vec<(u128, u64)> {
        let mem_wal = indices
            .iter()
            .find(|index| index.name == MEM_WAL_INDEX_NAME)
            .expect("the MemWAL index should be there");
        let details = load_mem_wal_index_details(mem_wal.clone()).unwrap();
        let mut progress = details
            .compacted_sstables
            .iter()
            .map(|sstable| (sstable.shard_id.as_u128(), sstable.generation))
            .collect::<Vec<_>>();
        progress.sort();
        progress
    }

    /// A MemWAL index recording no compaction progress yet, which recording
    /// progress requires the table to already have.
    fn empty_mem_wal_index() -> crate::format::IndexMetadata {
        new_mem_wal_index_meta(1, MemWalIndexDetails::default()).unwrap()
    }

    #[test]
    fn test_recording_progress_without_a_mem_wal_index_is_rejected() {
        let error = apply_with_indices(&backed_manifest(), vec![update(vec![(1, 7)])], Vec::new())
            .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("does not exist on this table"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_a_later_generation_supersedes_the_one_recorded_for_that_shard() {
        let manifest = backed_manifest();
        let (_, indices) = apply_with_indices(
            &manifest,
            vec![update(vec![(1, 3)])],
            vec![empty_mem_wal_index()],
        )
        .unwrap();
        let (_, indices) =
            apply_with_indices(&manifest, vec![update(vec![(1, 9)])], indices).unwrap();

        assert_eq!(progress(&indices), vec![(1, 9)]);
    }

    #[test]
    fn test_an_earlier_generation_is_rejected_rather_than_walking_a_shard_backwards() {
        let manifest = backed_manifest();
        let (_, indices) = apply_with_indices(
            &manifest,
            vec![update(vec![(1, 9)])],
            vec![empty_mem_wal_index()],
        )
        .unwrap();
        let error = apply_with_indices(&manifest, vec![update(vec![(1, 3)])], indices).unwrap_err();

        assert!(
            error.to_string().contains("Stale SSTable compaction"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_shards_are_tracked_independently() {
        let manifest = backed_manifest();
        let (_, indices) = apply_with_indices(
            &manifest,
            vec![update(vec![(1, 3)])],
            vec![empty_mem_wal_index()],
        )
        .unwrap();
        let (_, indices) =
            apply_with_indices(&manifest, vec![update(vec![(2, 5)])], indices).unwrap();

        assert_eq!(progress(&indices), vec![(1, 3), (2, 5)]);
    }

    #[test]
    fn test_recording_no_sstables_is_rejected() {
        let error = apply_with_indices(
            &backed_manifest(),
            vec![update(vec![])],
            vec![empty_mem_wal_index()],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("names no SSTable"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_recording_progress_leaves_other_indices_alone() {
        let kept = sample_index_metadata("by_a");
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![update(vec![(1, 7)])],
            vec![kept.clone(), empty_mem_wal_index()],
        )
        .unwrap();

        assert!(indices.iter().any(|index| index.uuid == kept.uuid));
    }

    #[test]
    fn test_two_writers_recording_progress_conflict() {
        let footprint = |actions| {
            Footprint::from(&CompositeOperation::new(vec![UserAction::new(
                "step", actions,
            )]))
        };

        let ours = footprint(vec![update(vec![(1, 3)])]);
        let theirs = footprint(vec![update(vec![(2, 5)])]);

        assert!(ours.conflicts_with(&theirs));
    }
    #[test]
    fn test_recording_progress_conflicts_with_rebuilding_the_memwal_index() {
        use crate::transaction::action::{AddIndexSegment, Ref};

        // Recording progress rewrites the whole MemWAL entry, so it collides
        // with a concurrent writer replacing that entry -- unlike a regular
        // index, where two writers may hold disjoint segments.
        let footprint = |actions| {
            Footprint::from(&CompositeOperation::new(vec![UserAction::new(
                "step", actions,
            )]))
        };

        let progress = footprint(vec![update(vec![(1, 3)])]);
        let rebuild = footprint(vec![Action::AddIndexSegment(AddIndexSegment {
            uuid: Uuid::from_u128(20),
            name: MEM_WAL_INDEX_NAME.into(),
            fields: vec![Ref::Committed(0)],
            index_details: None,
            index_version: 1,
            covered_fragments: None,
            files: Vec::new(),
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: true,
        })]);

        assert!(progress.conflicts_with(&rebuild));
        assert!(rebuild.conflicts_with(&progress));
    }
}
