// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Adjust the fragment coverage of an index segment.

use super::apply::ApplyState;
use super::proto::{non_empty, required};
use super::{Coordinate, Footprint, Ref};
use crate::format::pb;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use uuid::Uuid;

/// Adjust which fragments an existing index segment covers, without rewriting
/// the segment.
///
/// This is how a segment picks up fragments a compaction produced, or lets go
/// of ones it no longer describes, in cases where the index files themselves are
/// still good. Rewriting a segment's contents is a
/// [`RemoveIndexSegment`](super::RemoveIndexSegment) plus an
/// [`AddIndexSegment`](super::AddIndexSegment) instead.
///
/// Additions are applied before removals, so a fragment named on both sides ends
/// up outside the coverage.
///
/// # When a fragment may be added
///
/// Coverage says which fragments a segment's existing index files describe, so a
/// fragment may only be added when they already describe its rows. In practice
/// that means one case: a rewrite -- compaction, or an in-place column rewrite --
/// moved rows the segment already covered into a new fragment, and the segment
/// reaches them through the fragment-reuse remapping. The matching removal of the
/// old fragment ids belongs in the same action.
///
/// Adding a fragment of genuinely new rows is a writer error, even though nothing
/// here can detect it: the segment has no entries for those rows, so a query
/// served by the index would silently miss them. New rows are covered by building
/// a segment over them ([`AddIndexSegment`](super::AddIndexSegment)), not by
/// widening an existing one.
#[derive(Debug, Clone, PartialEq)]
pub struct AdjustIndexCoverage {
    pub uuid: Uuid,
    /// The logical index the segment belongs to. Conflict detection compares
    /// coverage across the segments of one index, so it needs the index and not
    /// just the segment.
    pub name: String,
    /// Fragments to bring into the coverage. A [`Ref::Local`] names one this
    /// operation minted, which is how a segment follows its rows into the
    /// fragment a rewrite in the same operation moved them to.
    pub add_fragments: Vec<Ref>,
    /// Committed fragment ids to drop from the coverage. Unlike the additions
    /// these take no [`Ref`]: a fragment minted in this same operation was not
    /// in the coverage to begin with.
    pub remove_fragments: Vec<u64>,
}

impl AdjustIndexCoverage {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let added = self
            .add_fragments
            .iter()
            .map(|fragment| self.coverage_id(state.resolve_fragment(*fragment)?))
            .collect::<Result<Vec<_>>>()?;
        let removed = self
            .remove_fragments
            .iter()
            .map(|fragment| self.coverage_id(*fragment))
            .collect::<Result<Vec<_>>>()?;

        let segment = state.index_segment_mut(self.uuid, &self.name)?;
        let Some(bitmap) = segment.fragment_bitmap.as_mut() else {
            return Err(Error::invalid_input(format!(
                "index segment {} records no fragment coverage, so there is nothing to adjust; \
                 rewrite the segment to give it coverage",
                self.uuid
            )));
        };
        for fragment in added {
            bitmap.insert(fragment);
        }
        for fragment in removed {
            bitmap.remove(fragment);
        }
        Ok(())
    }

    fn coverage_id(&self, fragment: u64) -> Result<u32> {
        u32::try_from(fragment).map_err(|_| {
            Error::invalid_input(format!(
                "AdjustIndexCoverage for segment {} names fragment {fragment}, which is beyond \
                 the largest fragment id an index can record ({})",
                self.uuid,
                u32::MAX
            ))
        })
    }

    /// Coverage says which fragments an index describes, never what any of them
    /// hold, so adjusting it cannot change a row a reader sees.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// The segment, by uuid, plus a claim on the fragments this brings under
    /// the index. The claim is what stops one writer widening a segment onto a
    /// fragment another writer is covering with a segment of its own.
    ///
    /// Removals claim nothing: coverage the index gives up cannot collide with
    /// coverage another writer takes on.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add(Coordinate::IndexSegment(self.uuid));
        footprint.extend_index_coverage(self.name.clone(), self.add_fragments.iter().copied());
    }
}

impl DeepSizeOf for AdjustIndexCoverage {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.uuid.as_bytes().deep_size_of_children(context)
            + self.name.deep_size_of_children(context)
            + self.add_fragments.deep_size_of_children(context)
            + self.remove_fragments.deep_size_of_children(context)
    }
}

impl From<&AdjustIndexCoverage> for pb::AdjustIndexCoverage {
    fn from(value: &AdjustIndexCoverage) -> Self {
        Self {
            uuid: Some((&value.uuid).into()),
            name: value.name.clone(),
            add_fragments: value
                .add_fragments
                .iter()
                .map(|fragment| (*fragment).into())
                .collect(),
            remove_fragments: value.remove_fragments.clone(),
        }
    }
}

impl TryFrom<pb::AdjustIndexCoverage> for AdjustIndexCoverage {
    type Error = Error;

    fn try_from(message: pb::AdjustIndexCoverage) -> Result<Self> {
        Ok(Self {
            uuid: Uuid::try_from(&required(message.uuid, "AdjustIndexCoverage.uuid")?)?,
            name: non_empty(message.name, "AdjustIndexCoverage.name")?,
            add_fragments: message
                .add_fragments
                .into_iter()
                .map(Ref::try_from)
                .collect::<Result<Vec<_>>>()?,
            remove_fragments: message.remove_fragments,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::IndexMetadata;
    use crate::transaction::action::test_support::{apply_with_indices, backed_manifest};
    use crate::transaction::action::{
        Action, AddFragment, AddIndexSegment, CompositeOperation, Footprint, UserAction,
    };
    use crate::transaction::test_support::sample_index_metadata;

    fn covering(name: &str, fragments: impl IntoIterator<Item = u32>) -> IndexMetadata {
        IndexMetadata {
            fragment_bitmap: Some(fragments.into_iter().collect()),
            ..sample_index_metadata(name)
        }
    }

    fn adjust(uuid: Uuid, name: &str, add: Vec<Ref>, remove: Vec<u64>) -> Action {
        Action::AdjustIndexCoverage(AdjustIndexCoverage {
            uuid,
            name: name.into(),
            add_fragments: add,
            remove_fragments: remove,
        })
    }

    #[test]
    fn test_adjust_index_coverage_adds_and_removes() {
        let segment = covering("by_a", [0, 1]);
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![adjust(
                segment.uuid,
                "by_a",
                vec![Ref::Committed(2)],
                vec![1],
            )],
            vec![segment],
        )
        .unwrap();

        assert_eq!(
            indices[0].fragment_bitmap,
            Some([0, 2].into_iter().collect())
        );
    }

    #[test]
    fn test_coverage_can_take_in_a_fragment_minted_in_the_same_operation() {
        // Appending and extending an index's reach over what was appended is
        // one operation, and the fragment has no committed id until apply.
        let segment = covering("by_a", [0]);
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![
                Action::AddFragment(AddFragment {
                    local: 0,
                    physical_rows: 10,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                adjust(segment.uuid, "by_a", vec![Ref::Local(0)], vec![]),
            ],
            vec![segment],
        )
        .unwrap();

        assert_eq!(
            indices[0].fragment_bitmap,
            Some([0, 1].into_iter().collect())
        );
    }

    #[test]
    fn test_a_fragment_named_on_both_sides_ends_up_uncovered() {
        let segment = covering("by_a", [0]);
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![adjust(
                segment.uuid,
                "by_a",
                vec![Ref::Committed(5)],
                vec![5],
            )],
            vec![segment],
        )
        .unwrap();

        assert_eq!(indices[0].fragment_bitmap, Some([0].into_iter().collect()));
    }

    #[test]
    fn test_adjusting_a_segment_that_is_not_there_is_rejected() {
        let error = apply_with_indices(
            &backed_manifest(),
            vec![adjust(
                Uuid::from_u128(1),
                "by_a",
                vec![Ref::Committed(0)],
                vec![],
            )],
            vec![covering("by_a", [0])],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("is not part of the dataset"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_adjusting_a_segment_with_no_recorded_coverage_is_rejected() {
        // Coverage of "unknown" is not an empty set to add to: turning it into
        // a concrete set would narrow what the segment serves, silently.
        let segment = IndexMetadata {
            fragment_bitmap: None,
            ..sample_index_metadata("system")
        };
        let error = apply_with_indices(
            &backed_manifest(),
            vec![adjust(
                segment.uuid,
                "system",
                vec![Ref::Committed(0)],
                vec![],
            )],
            vec![segment],
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("records no fragment coverage"),
            "unexpected error: {error}"
        );
    }

    fn footprint(actions: Vec<Action>) -> Footprint {
        Footprint::from(&CompositeOperation::new(vec![UserAction::new(
            "step", actions,
        )]))
    }

    fn build(name: &str, fragment: u64) -> Action {
        Action::AddIndexSegment(AddIndexSegment {
            uuid: Uuid::new_v4(),
            name: name.into(),
            fields: vec![Ref::Committed(0)],
            index_details: None,
            index_version: 1,
            covered_fragments: Some(vec![Ref::Committed(fragment)]),
            files: Vec::new(),
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: false,
        })
    }

    #[test]
    fn test_adjusting_a_segment_of_another_index_is_rejected() {
        let segment = covering("by_a", [0]);
        let error = apply_with_indices(
            &backed_manifest(),
            vec![adjust(
                segment.uuid,
                "by_b",
                vec![Ref::Committed(2)],
                vec![],
            )],
            vec![segment],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("belongs to index 'by_a'"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_two_writers_adjusting_the_same_segment_conflict() {
        let uuid = Uuid::from_u128(1);
        let ours = footprint(vec![adjust(uuid, "by_a", vec![Ref::Committed(1)], vec![])]);
        let same = footprint(vec![adjust(uuid, "by_a", vec![], vec![2])]);
        let other = footprint(vec![adjust(Uuid::from_u128(2), "by_a", vec![], vec![2])]);

        assert!(ours.conflicts_with(&same));
        assert!(!ours.conflicts_with(&other));
    }

    #[test]
    fn test_widening_onto_a_fragment_a_concurrent_segment_covers_conflicts() {
        let widen = footprint(vec![adjust(
            Uuid::from_u128(1),
            "by_a",
            vec![Ref::Committed(7)],
            vec![],
        )]);

        assert!(widen.conflicts_with(&footprint(vec![build("by_a", 7)])));
        assert!(!widen.conflicts_with(&footprint(vec![build("by_a", 8)])));
        assert!(!widen.conflicts_with(&footprint(vec![build("by_b", 7)])));
    }

    #[test]
    fn test_giving_up_coverage_claims_nothing() {
        // Coverage an index drops cannot collide with coverage another writer
        // takes on, so a removal-only adjustment leaves the index free.
        let narrow = footprint(vec![adjust(Uuid::from_u128(1), "by_a", vec![], vec![7])]);

        assert!(!narrow.conflicts_with(&footprint(vec![build("by_a", 7)])));
    }
}
