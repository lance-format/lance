// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Remove an index segment.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire, non_empty, required};
use super::{Coordinate, Footprint};
use crate::format::pb;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use uuid::Uuid;

/// Remove an index segment.
///
/// Dropping a whole logical index is one of these per segment carrying its
/// name, since the format knows only segments. Removing the last segment of an
/// index is what makes the index disappear.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoveIndexSegment {
    pub uuid: Uuid,
    /// The logical index the segment belongs to. Not needed to find the
    /// segment, which the uuid names outright; carried so apply can reject a
    /// removal that was planned against a different set of segments.
    pub name: String,
    pub data_change: bool,
}

impl RemoveIndexSegment {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        state.remove_index_segment(self.uuid, &self.name)
    }

    /// Left to the writer, for the same reason as
    /// [`AddIndexSegment`](super::AddIndexSegment): an index is normally derived
    /// state, but the MemWAL index holds rows a reader can see.
    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// The segment, by uuid. Only another action naming the same segment
    /// collides: a concurrent writer extending the same logical index is adding
    /// a segment of its own, which this removal leaves alone, and one that
    /// rebuilds what this segment covered leaves the index no worse off.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add(Coordinate::IndexSegment(self.uuid));
    }
}

impl DeepSizeOf for RemoveIndexSegment {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.uuid.as_bytes().deep_size_of_children(context)
            + self.name.deep_size_of_children(context)
    }
}

impl From<&RemoveIndexSegment> for pb::RemoveIndexSegment {
    fn from(value: &RemoveIndexSegment) -> Self {
        Self {
            uuid: Some((&value.uuid).into()),
            name: value.name.clone(),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb::RemoveIndexSegment> for RemoveIndexSegment {
    type Error = Error;

    fn try_from(message: pb::RemoveIndexSegment) -> Result<Self> {
        Ok(Self {
            uuid: Uuid::try_from(&required(message.uuid, "RemoveIndexSegment.uuid")?)?,
            name: non_empty(message.name, "RemoveIndexSegment.name")?,
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::action::test_support::{apply_with_indices, backed_manifest};
    use crate::transaction::action::{Action, CompositeOperation, Footprint, UserAction};
    use crate::transaction::test_support::sample_index_metadata;

    fn remove(uuid: Uuid, name: &str) -> Action {
        Action::RemoveIndexSegment(RemoveIndexSegment {
            uuid,
            name: name.into(),
            data_change: false,
        })
    }

    #[test]
    fn test_remove_index_segment_drops_only_the_segment_it_names() {
        let dropped = sample_index_metadata("by_a");
        let kept = sample_index_metadata("by_b");
        let dropped_uuid = dropped.uuid;

        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![remove(dropped_uuid, "by_a")],
            vec![dropped, kept.clone()],
        )
        .unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].uuid, kept.uuid);
    }

    #[test]
    fn test_dropping_an_index_removes_each_of_its_segments() {
        let first = sample_index_metadata("by_a");
        let second = sample_index_metadata("by_a");

        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![remove(first.uuid, "by_a"), remove(second.uuid, "by_a")],
            vec![first, second],
        )
        .unwrap();

        assert!(indices.is_empty());
    }

    #[test]
    fn test_removing_a_segment_that_is_not_there_is_rejected() {
        let error = apply_with_indices(
            &backed_manifest(),
            vec![remove(Uuid::from_u128(1), "by_a")],
            vec![sample_index_metadata("by_a")],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("is not part of the dataset"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_a_segment_can_be_replaced_within_one_operation() {
        use crate::transaction::action::{AddIndexSegment, Ref};

        let old = sample_index_metadata("by_a");
        let new_uuid = Uuid::from_u128(2);
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![
                remove(old.uuid, "by_a"),
                Action::AddIndexSegment(AddIndexSegment {
                    uuid: new_uuid,
                    name: "by_a".into(),
                    fields: vec![Ref::Committed(0)],
                    index_details: None,
                    index_version: 1,
                    covered_fragments: Some(vec![Ref::Committed(0)]),
                    files: Vec::new(),
                    base: None,
                    created_at: None,
                    dataset_version: None,
                    data_change: false,
                }),
            ],
            vec![old],
        )
        .unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].uuid, new_uuid);
    }

    #[test]
    fn test_removing_a_segment_of_another_index_is_rejected() {
        let segment = sample_index_metadata("by_a");
        let uuid = segment.uuid;
        let error = apply_with_indices(
            &backed_manifest(),
            vec![remove(uuid, "by_b")],
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
    fn test_removing_a_segment_leaves_a_concurrent_build_of_its_index_alone() {
        use crate::transaction::action::{AddIndexSegment, Ref};

        // Dropping a stale segment while someone else builds a fresh one is the
        // normal shape of index maintenance; the index ends up with the new
        // segment and without the old one either way round.
        let footprint = |actions| {
            Footprint::from(&CompositeOperation::new(vec![UserAction::new(
                "step", actions,
            )]))
        };
        let ours = footprint(vec![remove(Uuid::from_u128(1), "by_a")]);
        let theirs = footprint(vec![Action::AddIndexSegment(AddIndexSegment {
            uuid: Uuid::from_u128(2),
            name: "by_a".into(),
            fields: vec![Ref::Committed(0)],
            index_details: None,
            index_version: 1,
            covered_fragments: Some(vec![Ref::Committed(0)]),
            files: Vec::new(),
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: false,
        })]);

        assert!(!ours.conflicts_with(&theirs));
        assert!(!theirs.conflicts_with(&ours));
    }

    #[test]
    fn test_two_writers_removing_the_same_segment_conflict() {
        let footprint = |actions| {
            Footprint::from(&CompositeOperation::new(vec![UserAction::new(
                "step", actions,
            )]))
        };

        let uuid = Uuid::from_u128(1);
        let ours = footprint(vec![remove(uuid, "by_a")]);
        let same = footprint(vec![remove(uuid, "by_a")]);
        let other = footprint(vec![remove(Uuid::from_u128(2), "by_a")]);

        assert!(ours.conflicts_with(&same));
        assert!(!ours.conflicts_with(&other));
    }
}
