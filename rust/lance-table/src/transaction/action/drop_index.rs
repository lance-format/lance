// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Drop a logical index.

use super::Footprint;
use super::apply::ApplyState;
use super::proto::non_empty;
use crate::format::pb_actions;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Drop a logical index: every segment carrying `name`, as the segments stand
/// when the action applies.
///
/// Distinct from a [`RemoveIndexSegment`](super::RemoveIndexSegment) per
/// segment, which names the segments the writer read. A segment a concurrent
/// writer added under the name would survive that, and the dropped index would
/// come back holding only the new segment. This names the index instead, so it
/// removes whatever segments are there, and its footprint collides with any
/// concurrent claim on the index. `RemoveIndexSegment` stays the tool for
/// segment-level maintenance, such as an optimize replacing segments, which a
/// concurrent new segment must not be swept up in.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct DropIndex {
    /// The logical index to drop.
    pub name: String,
}

impl DropIndex {
    pub(super) fn apply(&self, _state: &mut ApplyState) -> Result<()> {
        Err(Error::not_supported("DropIndex is not implemented yet"))
    }

    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    pub(super) fn footprint(&self, _footprint: &mut Footprint) {}
}

impl From<&DropIndex> for pb_actions::DropIndex {
    fn from(value: &DropIndex) -> Self {
        Self {
            name: value.name.clone(),
        }
    }
}

impl TryFrom<pb_actions::DropIndex> for DropIndex {
    type Error = Error;

    fn try_from(message: pb_actions::DropIndex) -> Result<Self> {
        Ok(Self {
            name: non_empty(message.name, "DropIndex.name")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{DataFile, IndexMetadata};
    use crate::system_index::mem_wal::MEM_WAL_INDEX_NAME;
    use crate::transaction::action::test_support::{
        apply_with_indices, backed_manifest, footprint,
    };
    use crate::transaction::action::{
        Action, AddDataFile, AddFragment, AddIndexSegment, CompositeOperation, FieldRef, Ref,
        RemoveIndexSegment, UserStep,
    };
    use crate::transaction::test_support::{default_build_config, sample_index_metadata};
    use crate::transaction::{Operation, ReadVersionState, Transaction};
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;
    use uuid::Uuid;

    fn drop_index(name: &str) -> Action {
        Action::DropIndex(DropIndex { name: name.into() })
    }

    fn add_segment(name: &str, covered: Vec<Ref>) -> Action {
        Action::AddIndexSegment(AddIndexSegment {
            uuid: Uuid::new_v4(),
            name: name.into(),
            fields: vec![FieldRef::Committed(0)],
            covering_fields: Vec::new(),
            index_details: None,
            index_version: 1,
            covered_fragments: Some(covered),
            files: Vec::new(),
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: false,
        })
    }

    /// An append that indexes what it appended: the shape a concurrent writer
    /// keeping an index current takes.
    fn append_and_index(name: &str) -> Vec<Action> {
        vec![
            Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 10,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(0),
                file: DataFile::new_unstarted("data/new.lance", ConcreteFileVersion::V2_0),
                field_ids: vec![FieldRef::Committed(0)],
                data_change: true,
                base: None,
            }),
            add_segment(name, vec![Ref::Local(0)]),
        ]
    }

    fn segment_covering(name: &str, fragments: impl IntoIterator<Item = u32>) -> IndexMetadata {
        IndexMetadata {
            fragment_bitmap: Some(fragments.into_iter().collect()),
            ..sample_index_metadata(name)
        }
    }

    #[test]
    fn test_drop_index_removes_every_segment_of_the_index() {
        let kept = sample_index_metadata("by_b");
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![drop_index("by_a")],
            vec![
                segment_covering("by_a", [0]),
                segment_covering("by_a", []),
                kept.clone(),
            ],
        )
        .unwrap();

        assert_eq!(
            indices.iter().map(|index| index.uuid).collect::<Vec<_>>(),
            vec![kept.uuid]
        );
    }

    /// The point of naming the index rather than its segments: a segment
    /// committed after the drop read is dropped with the rest.
    #[test]
    fn test_drop_index_removes_a_segment_committed_after_the_read() {
        let read = backed_manifest();
        let mut current = read.clone();
        current.version += 1;
        let read_segment = segment_covering("by_a", [0]);
        let concurrent_segment = segment_covering("by_a", []);

        let (_, indices) = Transaction::new(
            read.version,
            Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
                "drop",
                vec![drop_index("by_a")],
            )])),
            None,
        )
        .build_manifest_with_read_version(
            Some(&current),
            vec![read_segment.clone(), concurrent_segment],
            "tx.txn",
            &default_build_config(),
            Some(ReadVersionState {
                manifest: &read,
                indices: &[read_segment],
            }),
        )
        .unwrap();

        assert!(indices.is_empty(), "{indices:?}");
    }

    #[test]
    fn test_dropping_an_index_that_is_not_there_is_rejected() {
        let error = apply_with_indices(
            &backed_manifest(),
            vec![drop_index("by_a")],
            vec![sample_index_metadata("by_b")],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("'by_a'"), "{error}");
    }

    /// A drop and a concurrent segment under the same name cannot both land:
    /// either the new segment resurrects the index, or the drop discards a
    /// segment its writer never saw. Both orders conflict, so the loser
    /// retries against the other's result.
    #[rstest]
    #[case::append_and_index_after_the_drop(append_and_index("by_a"), vec![drop_index("by_a")], true)]
    #[case::drop_after_append_and_index(vec![drop_index("by_a")], append_and_index("by_a"), true)]
    #[case::two_drops(vec![drop_index("by_a")], vec![drop_index("by_a")], true)]
    #[case::another_index(vec![drop_index("by_a")], append_and_index("by_b"), false)]
    #[case::unrelated_append(
        vec![drop_index("by_a")],
        append_and_index("by_b")[..2].to_vec(),
        false
    )]
    fn test_drop_index_conflicts(
        #[case] ours: Vec<Action>,
        #[case] committed: Vec<Action>,
        #[case] expected: bool,
    ) {
        assert_eq!(
            footprint(ours).conflicts_with(&footprint(committed)),
            expected
        );
    }

    /// An optimize replacing a segment is segment-level maintenance, and keeps
    /// committing alongside a writer that indexes what it appends.
    #[test]
    fn test_an_optimize_and_a_concurrent_append_and_index_do_not_conflict() {
        let optimize = footprint(vec![
            Action::RemoveIndexSegment(RemoveIndexSegment {
                uuid: Uuid::from_u128(1),
                name: "by_a".into(),
                data_change: false,
            }),
            add_segment("by_a", vec![Ref::Committed(0)]),
        ]);
        let append = footprint(append_and_index("by_a"));

        assert!(!optimize.conflicts_with(&append));
        assert!(!append.conflicts_with(&optimize));
    }

    /// Mirrors [`RemoveIndexSegment`]: an index is derived state, except the
    /// MemWAL index, which holds rows a reader can see.
    #[rstest]
    #[case::user_index("by_a", false)]
    #[case::mem_wal(MEM_WAL_INDEX_NAME, true)]
    fn test_drop_index_data_change(#[case] name: &str, #[case] expected: bool) {
        assert_eq!(drop_index(name).is_data_change(), expected);
    }
}
