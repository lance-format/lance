// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Adjust the fragment coverage of an index segment.

use super::apply::ApplyState;
use super::proto::{non_empty, required};
use super::{AddIndexSegment, Coordinate, FieldRef, Footprint, Ref};
use crate::format::pb_actions;
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
    /// The segment whose coverage changes.
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
    /// The fields the segment is keyed on, declared as
    /// [`AddIndexSegment::fields`](super::AddIndexSegment::fields) declares
    /// them. Apply rejects a declaration that does not match the segment.
    pub fields: Vec<FieldRef>,
    /// The fields whose values the segment carries, declared as
    /// [`AddIndexSegment::covering_fields`](super::AddIndexSegment::covering_fields)
    /// declares them.
    pub covering_fields: Vec<FieldRef>,
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
            .map(|fragment| self.coverage_id(state.resolve_fragment(Ref::Committed(*fragment))?))
            .collect::<Result<Vec<_>>>()?;
        let resolve = |fields: &[FieldRef]| {
            fields
                .iter()
                .map(|field| state.resolve_schema_field(*field, "AdjustIndexCoverage"))
                .collect::<Result<Vec<_>>>()
        };
        let (fields, covering_fields) = AddIndexSegment::lower_covering(
            &self.name,
            resolve(&self.fields)?,
            resolve(&self.covering_fields)?,
        )?;

        let segment = state.index_segment_mut(self.uuid, &self.name)?;
        if segment.fields != fields || segment.covering_fields != covering_fields {
            return Err(Error::invalid_input(format!(
                "AdjustIndexCoverage for segment {} of index '{}' declares fields {fields:?} \
                 (carried {covering_fields:?}), which does not match the segment's fields {:?} \
                 (carried {:?}); declare the segment's keyed and carried fields as an \
                 AddIndexSegment would",
                self.uuid, self.name, segment.fields, segment.covering_fields
            )));
        }
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
    /// fragment another writer is covering with a segment of its own. Those
    /// fragments are also required to still be there: coverage of a fragment a
    /// concurrent set removed describes rows the dataset no longer has.
    ///
    /// Each committed fragment brought in also requires the data of every
    /// keyed and carried field, as a new segment over it would: the widening
    /// holds only while those columns are the ones the remap was planned
    /// against. A rewrite committing first conflicts; one committing second
    /// prunes the fragment from the coverage as it applies. Overlays need no
    /// claim, since a read masks an overlay newer than the segment's
    /// `dataset_version`.
    ///
    /// Removals claim nothing: coverage the index gives up cannot collide with
    /// coverage another writer takes on.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.write(Coordinate::IndexSegment(self.uuid));
        footprint.extend_index_coverage(self.name.clone(), self.add_fragments.iter().copied());
        let dependencies = self.fields.iter().chain(self.covering_fields.iter());
        for fragment in self
            .add_fragments
            .iter()
            .filter_map(|fragment| fragment.committed())
        {
            footprint.require_fragment(fragment);
            footprint.require_field_data(fragment, dependencies.clone().copied());
        }
    }
}

impl DeepSizeOf for AdjustIndexCoverage {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.uuid.as_bytes().deep_size_of_children(context)
            + self.name.deep_size_of_children(context)
            + self.add_fragments.deep_size_of_children(context)
            + self.remove_fragments.deep_size_of_children(context)
            + self.fields.deep_size_of_children(context)
            + self.covering_fields.deep_size_of_children(context)
    }
}

impl From<&AdjustIndexCoverage> for pb_actions::AdjustIndexCoverage {
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
            fields: value.fields.iter().map(|field| (*field).into()).collect(),
            covering_fields: value
                .covering_fields
                .iter()
                .map(|field| (*field).into())
                .collect(),
        }
    }
}

impl TryFrom<pb_actions::AdjustIndexCoverage> for AdjustIndexCoverage {
    type Error = Error;

    fn try_from(message: pb_actions::AdjustIndexCoverage) -> Result<Self> {
        Ok(Self {
            uuid: Uuid::try_from(&required(message.uuid, "AdjustIndexCoverage.uuid")?)?,
            name: non_empty(message.name, "AdjustIndexCoverage.name")?,
            add_fragments: message
                .add_fragments
                .into_iter()
                .map(Ref::try_from)
                .collect::<Result<Vec<_>>>()?,
            remove_fragments: message.remove_fragments,
            fields: message
                .fields
                .into_iter()
                .map(FieldRef::try_from)
                .collect::<Result<Vec<_>>>()?,
            covering_fields: message
                .covering_fields
                .into_iter()
                .map(FieldRef::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{DataFile, IndexMetadata, Manifest};
    use crate::transaction::action::test_support::{
        apply_with_indices, backed_manifest, footprint,
    };
    use crate::transaction::action::{
        Action, AddDataFile, AddFragment, AddIndexSegment, FieldRef, RemoveFragment,
        TombstoneFieldData,
    };
    use crate::transaction::test_support::sample_index_metadata;
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;
    use std::sync::Arc;

    /// `backed_manifest` with fragment ids up to 9 handed out, so an
    /// adjustment can name fragments other than 0 as committed.
    fn manifest_with_reserved_ids() -> Manifest {
        let mut manifest = backed_manifest();
        manifest.max_fragment_id = Some(9);
        manifest
    }

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
            fields: vec![FieldRef::Committed(0)],
            covering_fields: Vec::new(),
        })
    }

    #[test]
    fn test_adjust_index_coverage_adds_and_removes() {
        let segment = covering("by_a", [0, 1]);
        let (_, indices) = apply_with_indices(
            &manifest_with_reserved_ids(),
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
                    id: Ref::Local(0),
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
            &manifest_with_reserved_ids(),
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

    /// Widening is held to the same rule as adding a segment: two segments of
    /// one index may not describe the same fragment.
    #[test]
    fn test_widening_onto_a_fragment_a_sibling_segment_covers_is_rejected() {
        let mut manifest = backed_manifest();
        let mut second = manifest.fragments[0].clone();
        second.id = 1;
        second.files[0].path = "data/1.lance".into();
        manifest.fragments = Arc::new(vec![manifest.fragments[0].clone(), second]);
        let widened = covering("by_a", [0]);
        let sibling = covering("by_a", [1]);
        let error = apply_with_indices(
            &manifest,
            vec![adjust(
                widened.uuid,
                "by_a",
                vec![Ref::Committed(1)],
                vec![],
            )],
            vec![widened, sibling],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        let message = error.to_string();
        assert!(message.contains("fragment 1"), "{message}");
        assert!(message.contains("counted twice"), "{message}");
    }

    #[rstest]
    #[case::added(vec![Ref::Committed(u64::from(u32::MAX) + 1)], vec![])]
    #[case::removed(vec![], vec![u64::from(u32::MAX) + 1])]
    fn test_a_fragment_id_no_manifest_can_hold_is_rejected(
        #[case] add: Vec<Ref>,
        #[case] remove: Vec<u64>,
    ) {
        let segment = covering("by_a", [0]);
        let error = apply_with_indices(
            &manifest_with_reserved_ids(),
            vec![adjust(segment.uuid, "by_a", add, remove)],
            vec![segment],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("fragment 4294967296"), "{error}");
    }

    #[test]
    fn test_adjusting_a_segment_that_is_not_there_is_rejected() {
        let error = apply_with_indices(
            &manifest_with_reserved_ids(),
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
            &manifest_with_reserved_ids(),
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

    fn build(name: &str, fragment: u64) -> Action {
        Action::AddIndexSegment(AddIndexSegment {
            uuid: Uuid::new_v4(),
            name: name.into(),
            fields: vec![FieldRef::Committed(0)],
            covering_fields: Vec::new(),
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
            &manifest_with_reserved_ids(),
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
    fn test_widening_onto_a_fragment_a_concurrent_set_removes_conflicts() {
        let widen = footprint(vec![adjust(
            Uuid::from_u128(1),
            "by_a",
            vec![Ref::Committed(7)],
            vec![],
        )]);
        let remove = |fragment: u64| {
            footprint(vec![Action::RemoveFragment(RemoveFragment {
                fragment: Ref::Committed(fragment),
                data_change: false,
            })])
        };

        // Either order: coverage of a fragment that is gone describes rows the
        // dataset no longer has.
        assert!(widen.conflicts_with(&remove(7)));
        assert!(remove(7).conflicts_with(&widen));
        assert!(!widen.conflicts_with(&remove(8)));
    }

    #[test]
    fn test_giving_up_coverage_claims_nothing() {
        // Coverage an index drops cannot collide with coverage another writer
        // takes on, so a removal-only adjustment leaves the index free.
        let narrow = footprint(vec![adjust(Uuid::from_u128(1), "by_a", vec![], vec![7])]);

        assert!(!narrow.conflicts_with(&footprint(vec![build("by_a", 7)])));
    }

    fn widen_declaring(fields: Vec<i32>, covering_fields: Vec<i32>, onto: u64) -> Action {
        Action::AdjustIndexCoverage(AdjustIndexCoverage {
            uuid: Uuid::from_u128(1),
            name: "by_a".into(),
            add_fragments: vec![Ref::Committed(onto)],
            remove_fragments: Vec::new(),
            fields: fields.into_iter().map(FieldRef::Committed).collect(),
            covering_fields: covering_fields
                .into_iter()
                .map(FieldRef::Committed)
                .collect(),
        })
    }

    fn rewrite(fragment: u64, field: i32) -> Vec<Action> {
        vec![
            Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(fragment),
                field_ids: vec![FieldRef::Committed(field)],
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(fragment),
                file: DataFile::new_unstarted("data/rewritten.lance", ConcreteFileVersion::V2_0),
                field_ids: vec![FieldRef::Committed(field)],
                data_change: true,
                base: None,
            }),
        ]
    }

    /// Widening is a claim that the segment's files already describe the
    /// fragment's rows, which only holds while the indexed columns there are
    /// the ones the remap was planned against. A rewrite of a keyed or carried
    /// column that commits first breaks that, the same as it would for a new
    /// segment over the fragment. One that commits second needs no conflict:
    /// it prunes the fragment from the widened coverage as it applies.
    #[rstest]
    #[case::keyed_column_rewritten_first(vec![0], vec![1], 0, true, true)]
    #[case::carried_column_rewritten_first(vec![0], vec![1], 1, true, true)]
    #[case::other_column_rewritten_first(vec![0], vec![1], 2, true, false)]
    #[case::keyed_column_rewritten_second(vec![0], vec![1], 0, false, false)]
    fn test_widening_requires_the_data_it_covers(
        #[case] fields: Vec<i32>,
        #[case] covering_fields: Vec<i32>,
        #[case] rewritten: i32,
        #[case] rewrite_commits_first: bool,
        #[case] expected: bool,
    ) {
        let widen = footprint(vec![widen_declaring(fields, covering_fields, 7)]);
        let rewrite = footprint(rewrite(7, rewritten));

        let conflicts = if rewrite_commits_first {
            widen.conflicts_with(&rewrite)
        } else {
            rewrite.conflicts_with(&widen)
        };
        assert_eq!(conflicts, expected);
    }

    /// The order the footprint lets through: the rewrite lands on a version
    /// where the widened segment already covers the fragment, and takes it
    /// back out.
    #[test]
    fn test_a_rewrite_after_a_widening_prunes_the_widened_fragment() {
        let segment = covering("by_a", [0, 1]);
        let mut manifest = backed_manifest();
        let mut second = manifest.fragments[0].clone();
        second.id = 1;
        second.files[0].path = "data/1.lance".into();
        manifest.fragments = Arc::new(vec![manifest.fragments[0].clone(), second]);

        let (_, indices) = apply_with_indices(&manifest, rewrite(1, 0), vec![segment]).unwrap();

        assert_eq!(indices[0].fragment_bitmap, Some([0].into_iter().collect()));
    }

    /// The declared fields are what the footprint requires, so they have to be
    /// the segment's own. They are declared the way a segment is added, keys
    /// and carried columns apart, and compared with the legacy form the
    /// manifest stores.
    #[rstest]
    #[case::matching(vec![0], vec![], vec![0], vec![], None)]
    #[case::matching_with_a_carried_column(vec![0, 1], vec![1], vec![0], vec![1], None)]
    #[case::different_key(vec![0], vec![], vec![1], vec![], Some("fields"))]
    #[case::missing_key(vec![0], vec![], vec![], vec![], Some("fields"))]
    #[case::undeclared_carried_column(vec![0, 1], vec![1], vec![0, 1], vec![], Some("fields"))]
    #[case::carried_column_declared_as_a_key(vec![0, 1], vec![1], vec![0], vec![], Some("fields"))]
    fn test_the_declared_fields_must_match_the_segment(
        #[case] segment_fields: Vec<i32>,
        #[case] segment_covering_fields: Vec<i32>,
        #[case] fields: Vec<i32>,
        #[case] covering_fields: Vec<i32>,
        #[case] expected_error: Option<&str>,
    ) {
        let mut manifest = manifest_with_reserved_ids();
        manifest.schema.fields.push({
            let mut carried = crate::transaction::action::test_support::added_field("carried");
            carried.id = 1;
            carried
        });
        let segment = IndexMetadata {
            uuid: Uuid::from_u128(1),
            fields: segment_fields,
            covering_fields: segment_covering_fields,
            ..covering("by_a", [0])
        };

        let result = apply_with_indices(
            &manifest,
            vec![widen_declaring(fields, covering_fields, 2)],
            vec![segment],
        );

        let Some(expected_error) = expected_error else {
            let (_, indices) = result.unwrap();
            assert_eq!(
                indices[0].fragment_bitmap,
                Some([0, 2].into_iter().collect())
            );
            return;
        };
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        let message = error.to_string();
        assert!(message.contains(expected_error), "{message}");
        assert!(message.contains("does not match"), "{message}");
    }
}
