// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Drop a logical index.

use super::Footprint;
use super::apply::{ApplyState, reject_frag_reuse_history};
use super::proto::non_empty;
use crate::format::pb_actions;
use crate::system_index::mem_wal::MEM_WAL_INDEX_NAME;
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
    name: String,
}

impl DropIndex {
    /// Drop the logical index `name`.
    ///
    /// # Errors
    ///
    /// Rejects an empty `name`.
    pub fn try_new(name: impl Into<String>) -> Result<Self> {
        Ok(Self {
            name: non_empty(name.into(), "DropIndex.name")?,
        })
    }

    /// The logical index dropped.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        reject_frag_reuse_history("DropIndex", &self.name)?;
        state.drop_index(&self.name)
    }

    /// An index is derived state, so dropping one changes no row a reader
    /// sees -- except the MemWAL index, which holds rows of its own. Unlike
    /// [`RemoveIndexSegment`](super::RemoveIndexSegment) there is no flag to
    /// leave this to the writer: a whole index is never dropped as part of a
    /// rearrangement.
    pub(super) fn is_data_change(&self) -> bool {
        self.name == MEM_WAL_INDEX_NAME
    }

    /// A claim on the index with unstated coverage, which collides with every
    /// other claim on the same name, in either order. Nothing else is written:
    /// the segments it removes are whatever the index holds at apply.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.rewrite_index(self.name.clone());
    }
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
        Self::try_new(message.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::IndexMetadata;
    use crate::transaction::action::test_support::{apply_with_indices, backed_manifest};
    use crate::transaction::action::{Action, CompositeOperation, UserStep};
    use crate::transaction::test_support::{default_build_config, sample_index_metadata};
    use crate::transaction::{Operation, ReadVersionState, Transaction};

    use rstest::rstest;

    fn drop_index(name: &str) -> Action {
        Action::DropIndex(DropIndex::try_new(name).unwrap())
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

    /// Mirrors [`RemoveIndexSegment`]: an index is derived state, except the
    /// MemWAL index, which holds rows a reader can see.
    #[rstest]
    #[case::user_index("by_a", false)]
    #[case::mem_wal(MEM_WAL_INDEX_NAME, true)]
    fn test_drop_index_data_change(#[case] name: &str, #[case] expected: bool) {
        assert_eq!(drop_index(name).is_data_change(), expected);
    }
}
