// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Record a compaction's row moves in the fragment reuse history.

use super::apply::ApplyState;
use super::proto::required;
use super::{Footprint, Ref};
use crate::format::pb::fragment_reuse_index_details as pb_fri;
use crate::format::pb_actions;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};

/// The mapping of an [`AppendFragReuseTransition`]: how the surviving source
/// rows are laid out across the destinations.
pub type FragReuseMapping = pb_fri::transition::Mapping;

/// Record that a compaction in the same operation moved rows from `sources`
/// to `destinations`, so index segments built over the sources keep serving
/// those rows by translating through the fragment reuse history.
///
/// The history is not an index on the action path, and this action never
/// names its entry. Apply appends the transition to the history as it stands
/// on the version being committed on top of, at every attempt, deriving the
/// digests itself: a source's from the fragment as it stands, a destination's
/// from the [`AddFragment`](super::AddFragment) that adds it.
///
/// Apply rejects the transition unless every source is removed in the same
/// operation by a [`RemoveFragment`](super::RemoveFragment) with
/// `data_change` false; every destination is added in the same operation and
/// left without a deletion file; the mapping conserves rows per source and in
/// total; no other transition consumes a source; no index stores stable row
/// ids; and a stable partition has between 1 and 65,536 destinations.
///
/// Recording a transition is never required: a compaction that moves covered
/// rows without one is legal, and the indices lose that coverage.
#[derive(Debug, Clone, PartialEq)]
pub struct AppendFragReuseTransition {
    /// The fragments the rows were read from, in scan order.
    pub sources: Vec<Ref>,
    /// The fragments the rows were written to, in output (or label) order.
    pub destinations: Vec<Ref>,
    pub mapping: FragReuseMapping,
}

impl AppendFragReuseTransition {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let resolve = |refs: &[Ref]| {
            refs.iter()
                .map(|fragment| state.resolve_fragment(*fragment))
                .collect::<Result<Vec<_>>>()
        };
        let sources = resolve(&self.sources)?;
        let destinations = resolve(&self.destinations)?;
        if sources.is_empty() || destinations.is_empty() {
            return Err(Error::invalid_input(format!(
                "AppendFragReuseTransition needs at least one source and one destination, got \
                 sources {sources:?} and destinations {destinations:?}"
            )));
        }
        state.stage_frag_reuse_transition(sources, destinations, self.mapping.clone());
        Ok(())
    }

    /// Moving rows changes none of their values.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// Nothing of its own. The sources are removed by `RemoveFragment`s in the
    /// same operation, whose claims already make two compactions of one
    /// fragment, or a compaction and a deletion from one of its sources,
    /// collide; transitions with disjoint sources commute.
    pub(super) fn footprint(&self, _footprint: &mut Footprint) {}
}

impl DeepSizeOf for AppendFragReuseTransition {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        let mapping = match &self.mapping {
            FragReuseMapping::OrderedCompaction(ordered) => ordered.changed_row_addrs.capacity(),
            FragReuseMapping::StablePartition(partition) => partition.map_id.capacity(),
        };
        self.sources.deep_size_of_children(context)
            + self.destinations.deep_size_of_children(context)
            + mapping
    }
}

impl From<&AppendFragReuseTransition> for pb_actions::AppendFragReuseTransition {
    fn from(value: &AppendFragReuseTransition) -> Self {
        use pb_actions::append_frag_reuse_transition::Mapping;
        Self {
            sources: value.sources.iter().map(|id| (*id).into()).collect(),
            destinations: value.destinations.iter().map(|id| (*id).into()).collect(),
            mapping: Some(match &value.mapping {
                FragReuseMapping::OrderedCompaction(ordered) => {
                    Mapping::OrderedCompaction(ordered.clone())
                }
                FragReuseMapping::StablePartition(partition) => {
                    Mapping::StablePartition(partition.clone())
                }
            }),
        }
    }
}

impl TryFrom<pb_actions::AppendFragReuseTransition> for AppendFragReuseTransition {
    type Error = Error;

    fn try_from(message: pb_actions::AppendFragReuseTransition) -> Result<Self> {
        use pb_actions::append_frag_reuse_transition::Mapping;
        let refs = |refs: Vec<pb_actions::Ref>| {
            refs.into_iter()
                .map(Ref::try_from)
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            sources: refs(message.sources)?,
            destinations: refs(message.destinations)?,
            mapping: match required(message.mapping, "AppendFragReuseTransition.mapping")? {
                Mapping::OrderedCompaction(ordered) => FragReuseMapping::OrderedCompaction(ordered),
                Mapping::StablePartition(partition) => FragReuseMapping::StablePartition(partition),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_flags::{FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS, FLAG_FRAGMENT_REUSE_INDEX};
    use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
    use crate::format::{DataFile, DeletionFile, DeletionFileType, IndexMetadata, Manifest};
    use crate::transaction::action::test_support::frag_reuse::{
        ROWS, compaction, digest, history, ordered, read_history, recorded, stable_partition,
        survivors, table, transition,
    };
    use crate::transaction::action::test_support::{apply_with_indices, footprint};
    use crate::transaction::action::{Action, RemoveFragment, SetDeletionFile};
    use crate::transaction::test_support::sample_index_metadata;
    use rstest::rstest;
    use std::sync::Arc;

    fn commit(
        manifest: &Manifest,
        actions: Vec<Action>,
        indices: Vec<IndexMetadata>,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        apply_with_indices(manifest, actions, indices)
    }

    /// The first transition creates the history, tagged, and marks the
    /// manifest as carrying one; a table with stable row ids is marked as
    /// carrying both.
    #[rstest]
    #[case::row_addresses(false)]
    #[case::stable_row_ids(true)]
    fn test_the_first_transition_creates_a_tagged_history(#[case] stable_row_ids: bool) {
        let manifest = table(3, stable_row_ids);
        let (next, indices) = commit(
            &manifest,
            compaction(
                &[0, 1],
                0,
                stable_row_ids,
                Some(ordered(survivors(&[0, 1], &[]))),
            ),
            Vec::new(),
        )
        .unwrap();

        let (index_version, legacy_versions, transitions) = read_history(&indices).unwrap();
        assert_eq!(index_version, 1);
        assert!(legacy_versions.is_empty());
        assert_eq!(transitions.len(), 1);
        let destination = next.max_fragment_id.unwrap() as u64;
        assert_eq!(
            transitions[0].sources,
            vec![digest(0, ROWS), digest(1, ROWS)]
        );
        assert_eq!(
            transitions[0].destinations,
            vec![digest(destination, 2 * ROWS)]
        );
        for flags in [next.reader_feature_flags, next.writer_feature_flags] {
            assert_ne!(flags & FLAG_FRAGMENT_REUSE_INDEX, 0);
            assert_eq!(
                flags & FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS != 0,
                stable_row_ids
            );
        }
    }

    /// A version 0 history becomes version 1 with its legacy records kept as
    /// they were, and the new transition appended after them.
    #[test]
    fn test_a_transition_lifts_a_v0_history_keeping_its_legacy_versions() {
        let legacy = pb_fri::Version {
            dataset_version: 1,
            groups: vec![pb_fri::Group {
                changed_row_addrs: survivors(&[7], &[]),
                old_fragments: vec![digest(7, ROWS)],
                new_fragments: vec![digest(8, ROWS)],
            }],
        };
        let entry = history(0, vec![legacy.clone()], Vec::new());
        let (_, indices) = commit(
            &table(3, false),
            compaction(&[0], 0, false, Some(ordered(survivors(&[0], &[])))),
            vec![entry.clone()],
        )
        .unwrap();

        let (index_version, legacy_versions, transitions) = read_history(&indices).unwrap();
        assert_eq!(index_version, 1);
        assert_eq!(legacy_versions, vec![legacy]);
        assert_eq!(transitions.len(), 1);
        assert!(indices.iter().all(|index| index.uuid != entry.uuid));
    }

    /// The transition is appended to the history the version being committed
    /// on top of carries, after whatever it already records.
    #[test]
    fn test_a_transition_appends_to_the_current_history() {
        let entry = history(1, Vec::new(), vec![recorded(&[7], &[8])]);
        let (_, indices) = commit(
            &table(3, false),
            compaction(&[0], 0, false, Some(ordered(survivors(&[0], &[])))),
            vec![entry],
        )
        .unwrap();

        let (_, _, transitions) = read_history(&indices).unwrap();
        assert_eq!(transitions.len(), 2);
        assert_eq!(transitions[0], recorded(&[7], &[8]));
        assert_eq!(transitions[1].sources, vec![digest(0, ROWS)]);
    }

    /// A deletion the source carries is part of its digest, and the bitmap
    /// must leave the deleted rows out.
    #[test]
    fn test_a_source_digest_counts_its_deleted_rows() {
        let mut manifest = table(3, false);
        Arc::make_mut(&mut manifest.fragments)[0].deletion_file = Some(DeletionFile {
            read_version: 1,
            id: 1,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(1),
            base_id: None,
        });
        let mut actions = compaction(&[0], 0, false, Some(ordered(survivors(&[0], &[(0, 3)]))));
        if let Action::AddFragment(add) = &mut actions[1] {
            add.physical_rows = ROWS - 1;
        }

        let (_, indices) = commit(&manifest, actions, Vec::new()).unwrap();

        let (_, _, transitions) = read_history(&indices).unwrap();
        assert_eq!(
            transitions[0].sources,
            vec![pb_fri::FragmentDigest {
                id: 0,
                physical_rows: ROWS,
                num_deleted_rows: 1,
            }]
        );
    }

    fn reserved_destination(manifest: &mut Manifest) -> u64 {
        let reserved = manifest.max_fragment_id.unwrap() as u64 + 1;
        manifest.max_fragment_id = Some(reserved as u32);
        reserved
    }

    /// One case per check apply runs on a transition.
    #[rstest]
    #[case::source_not_removed(
        |_: &mut Manifest, actions: &mut Vec<Action>| { actions.remove(1); },
        "does not remove"
    )]
    #[case::source_removed_as_a_data_change(
        |_: &mut Manifest, actions: &mut Vec<Action>| {
            actions[1] = Action::RemoveFragment(RemoveFragment {
                fragment: Ref::Committed(1),
                data_change: true,
            });
        },
        "data_change true"
    )]
    #[case::destination_not_added(
        |_: &mut Manifest, actions: &mut Vec<Action>| {
            actions.push(transition(&[0], &[Ref::Committed(2)], ordered(survivors(&[0], &[]))));
            actions.retain(|action| !matches!(
                action,
                Action::AppendFragReuseTransition(t) if t.sources.len() == 2
            ));
        },
        "does not add"
    )]
    #[case::destination_with_a_deletion_file(
        |manifest: &mut Manifest, actions: &mut Vec<Action>| {
            let reserved = reserved_destination(manifest);
            if let Action::AddFragment(add) = &mut actions[2] {
                add.id = Ref::Committed(reserved);
            }
            if let Action::AddDataFile(file) = &mut actions[3] {
                file.fragment = Ref::Committed(reserved);
            }
            if let Action::AppendFragReuseTransition(t) = &mut actions[4] {
                t.destinations = vec![Ref::Committed(reserved)];
            }
            actions.push(Action::SetDeletionFile(SetDeletionFile {
                fragment: reserved,
                deletion_file: Some(DeletionFile {
                    read_version: 1,
                    id: 1,
                    file_type: DeletionFileType::Array,
                    num_deleted_rows: Some(1),
                    base_id: None,
                }),
                data_change: true,
                base: None,
            }));
        },
        "carries a deletion file"
    )]
    #[case::bitmap_missing_a_surviving_row(
        |_: &mut Manifest, actions: &mut Vec<Action>| {
            if let Action::AppendFragReuseTransition(t) = &mut actions[4] {
                t.mapping = ordered(survivors(&[0, 1], &[(1, 4)]));
            }
        },
        "surviving rows in fragment 1"
    )]
    #[case::bitmap_marking_rows_outside_the_sources(
        |_: &mut Manifest, actions: &mut Vec<Action>| {
            if let Action::AppendFragReuseTransition(t) = &mut actions[4] {
                t.mapping = ordered(survivors(&[0, 1, 2], &[]));
            }
        },
        "outside its sources"
    )]
    #[case::rows_not_conserved(
        |_: &mut Manifest, actions: &mut Vec<Action>| {
            if let Action::AddFragment(add) = &mut actions[2] {
                add.physical_rows = 2 * ROWS - 1;
            }
        },
        "does not conserve rows"
    )]
    #[case::source_consumed_twice_in_one_operation(
        |_: &mut Manifest, actions: &mut Vec<Action>| {
            actions.push(transition(&[1], &[Ref::Local(0)], ordered(survivors(&[1], &[]))));
        },
        "already consumes"
    )]
    #[case::source_consumed_by_the_history(
        |manifest: &mut Manifest, _: &mut Vec<Action>| {
            manifest.version = 7;
        },
        "already consumes"
    )]
    #[case::stable_partition_without_a_destination(
        |_: &mut Manifest, actions: &mut Vec<Action>| {
            if let Action::AppendFragReuseTransition(t) = &mut actions[4] {
                t.mapping = stable_partition();
                t.destinations.clear();
            }
        },
        "at least one source and one destination"
    )]
    #[case::source_carrying_an_overlay(
        |manifest: &mut Manifest, _: &mut Vec<Action>| {
            Arc::make_mut(&mut manifest.fragments)[0].overlays.push(DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("o.lance", vec![0], None),
                coverage: OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([0u32])),
                committed_version: 1,
            });
        },
        "data overlay files"
    )]
    fn test_a_transition_apply_rejects(
        #[case] change: fn(&mut Manifest, &mut Vec<Action>),
        #[case] expected: &str,
    ) {
        let mut manifest = table(3, false);
        let mut actions = compaction(&[0, 1], 0, false, Some(ordered(survivors(&[0, 1], &[]))));
        change(&mut manifest, &mut actions);
        // The marker case: the history already records a transition out of
        // fragment 1.
        let indices = if manifest.version == 7 {
            manifest.version = 1;
            vec![history(1, Vec::new(), vec![recorded(&[1], &[9])])]
        } else {
            Vec::new()
        };

        let error = commit(&manifest, actions, indices).unwrap_err();

        assert!(
            matches!(
                error,
                lance_core::Error::InvalidInput { .. } | lance_core::Error::NotSupported { .. }
            ),
            "{error:?}"
        );
        assert!(error.to_string().contains(expected), "{error}");
    }

    /// The history translates row addresses, so it cannot record moves on a
    /// table holding an index of stable row ids.
    #[test]
    fn test_a_transition_is_rejected_beside_a_row_id_index() {
        let manifest = table(3, true);
        let mut row_id_index = sample_index_metadata("by_id");
        row_id_index.index_details = None;

        let error = commit(
            &manifest,
            compaction(&[0], 0, true, Some(ordered(survivors(&[0], &[])))),
            vec![row_id_index],
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("stores stable row ids"),
            "{error}"
        );
    }

    /// A compaction may move covered rows without recording a transition:
    /// the indices lose that coverage, which is legal.
    #[test]
    fn test_a_compaction_without_a_transition_is_legal() {
        let segment = sample_index_metadata("by_id");
        let (next, indices) = commit(
            &table(3, false),
            compaction(&[0], 0, false, None),
            vec![segment.clone()],
        )
        .unwrap();

        assert!(read_history(&indices).is_none());
        assert!(next.fragments.iter().all(|fragment| fragment.id != 0));
        assert!(indices.iter().any(|index| index.uuid == segment.uuid));
    }

    /// The sources' `RemoveFragment`s carry the conflict: two compactions of
    /// one fragment collide, and compactions of disjoint fragments commute.
    #[rstest]
    #[case::disjoint_sources(&[0], &[1], false)]
    #[case::shared_source(&[0, 1], &[1, 2], true)]
    fn test_two_compactions_recording_transitions(
        #[case] ours: &[u64],
        #[case] theirs: &[u64],
        #[case] expected: bool,
    ) {
        let mapping = |sources: &[u64]| Some(ordered(survivors(sources, &[])));
        let ours = footprint(compaction(ours, 0, false, mapping(ours)));
        let theirs = footprint(compaction(theirs, 0, false, mapping(theirs)));
        assert_eq!(ours.conflicts_with(&theirs), expected);
        assert_eq!(theirs.conflicts_with(&ours), expected);
    }

    #[test]
    fn test_round_trips_through_the_wire() {
        for mapping in [ordered(survivors(&[0], &[])), stable_partition()] {
            let action = AppendFragReuseTransition {
                sources: vec![Ref::Committed(0)],
                destinations: vec![Ref::Local(0)],
                mapping,
            };
            let decoded = AppendFragReuseTransition::try_from(
                pb_actions::AppendFragReuseTransition::from(&action),
            )
            .unwrap();
            assert_eq!(decoded, action);
        }
    }
}
