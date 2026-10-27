// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Trim the fragment reuse history.

use super::Footprint;
use super::apply::ApplyState;
use crate::format::pb_actions;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Drop every transition of the fragment reuse history that no index still
/// needs.
///
/// Declarative: it names no transition. Apply decides what to drop against
/// the history and the index list as this operation leaves them, at every
/// attempt, so a transition a concurrent writer appended is judged afresh
/// rather than spliced away. It only drops what is safe to drop, so it never
/// conflicts. A history left with no record is removed.
#[derive(Debug, Clone, Default, PartialEq, DeepSizeOf)]
pub struct TrimFragReuse {}

impl TrimFragReuse {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        state.request_frag_reuse_trim();
        Ok(())
    }

    /// Dropping a translation nobody uses changes no row a reader sees.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// Nothing: it never conflicts.
    pub(super) fn footprint(&self, _footprint: &mut Footprint) {}
}

impl From<&TrimFragReuse> for pb_actions::TrimFragReuse {
    fn from(_value: &TrimFragReuse) -> Self {
        Self {}
    }
}

impl TryFrom<pb_actions::TrimFragReuse> for TrimFragReuse {
    type Error = Error;

    fn try_from(_message: pb_actions::TrimFragReuse) -> Result<Self> {
        Ok(Self {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::format::IndexMetadata;
    use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
    use crate::transaction::action::test_support::frag_reuse::{
        compaction, history, ordered, read_history, recorded, survivors, table,
    };
    use crate::transaction::action::test_support::{apply_with_indices, footprint};
    use crate::transaction::action::{
        Action, AddDataFile, AddOverlays, FieldRef, Ref, TombstoneFieldData,
    };
    use crate::transaction::test_support::sample_index_metadata;
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;

    fn trim() -> Action {
        Action::TrimFragReuse(TrimFragReuse {})
    }

    fn segment(name: &str, provenance: &[u32]) -> IndexMetadata {
        let mut segment = sample_index_metadata(name);
        segment.fragment_bitmap = Some(provenance.iter().copied().collect());
        segment
    }

    fn sources(
        transitions: &[crate::format::pb::fragment_reuse_index_details::Transition],
    ) -> Vec<Vec<u64>> {
        transitions
            .iter()
            .map(|transition| transition.sources.iter().map(|digest| digest.id).collect())
            .collect()
    }

    /// A transition is needed while some index holds addresses it moves (its
    /// provenance names a source) and has not re-covered every destination
    /// directly. The trim keeps exactly those, and removes a history left
    /// with none. The history moves 7 into 0 and 8 into 1.
    #[rstest]
    #[case::needed_by_a_translating_segment(vec![segment("a", &[7])], Some(vec![vec![7]]))]
    #[case::caught_up_by_a_direct_segment(vec![segment("a", &[0])], None)]
    #[case::partially_caught_up(vec![segment("a", &[7]), segment("b", &[1])], Some(vec![vec![7]]))]
    #[case::no_index(Vec::new(), None)]
    fn test_a_trim_keeps_what_an_index_needs(
        #[case] segments: Vec<IndexMetadata>,
        #[case] kept: Option<Vec<Vec<u64>>>,
    ) {
        let entry = history(
            1,
            Vec::new(),
            vec![recorded(&[7], &[0]), recorded(&[8], &[1])],
        );
        let mut indices = segments;
        indices.push(entry);

        let (_, indices) = apply_with_indices(&table(3, false), vec![trim()], indices).unwrap();

        assert_eq!(
            read_history(&indices).map(|(_, _, transitions)| sources(&transitions)),
            kept
        );
    }

    /// With nothing to drop the entry stands as it is, uuid and all.
    #[test]
    fn test_a_trim_with_nothing_to_drop_leaves_the_entry() {
        let entry = history(1, Vec::new(), vec![recorded(&[7], &[0])]);
        let (_, indices) = apply_with_indices(
            &table(3, false),
            vec![trim()],
            vec![segment("a", &[7]), entry.clone()],
        )
        .unwrap();

        assert!(indices.iter().any(|index| index.uuid == entry.uuid));
    }

    /// The trim is judged on the history it lands on: a transition a
    /// concurrent compaction appended, still needed, survives a trim planned
    /// before it existed. Neither names anything the other claims.
    #[test]
    fn test_a_trim_keeps_a_transition_appended_concurrently() {
        let appended = compaction(&[0], 0, false, Some(ordered(survivors(&[0], &[]))));
        let (landed, indices) =
            apply_with_indices(&table(3, false), appended.clone(), vec![segment("a", &[0])])
                .unwrap();

        let (_, indices) = apply_with_indices(&landed, vec![trim()], indices).unwrap();

        let (_, _, transitions) = read_history(&indices).unwrap();
        assert_eq!(sources(&transitions), vec![vec![0]]);
        assert!(!footprint(vec![trim()]).conflicts_with(&footprint(appended.clone())));
        assert!(!footprint(appended).conflicts_with(&footprint(vec![trim()])));
    }

    /// An in-place rewrite of a fragment that a segment reaches through the
    /// history withdraws that segment's provenance, so the stale entries it
    /// holds for the rewritten rows are no longer translated there.
    #[test]
    fn test_an_in_place_rewrite_withdraws_coverage_along_the_history() {
        let entry = history(1, Vec::new(), vec![recorded(&[7], &[0])]);
        let translating = segment("a", &[7]);
        let (_, indices) = apply_with_indices(
            &table(3, false),
            vec![
                Action::TombstoneFieldData(TombstoneFieldData {
                    fragment: Ref::Committed(0),
                    field_ids: vec![FieldRef::Committed(0)],
                    data_change: true,
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Committed(0),
                    file: DataFile::new_unstarted("data/new.lance", ConcreteFileVersion::V2_0),
                    field_ids: vec![FieldRef::Committed(0)],
                    data_change: true,
                    base: None,
                }),
            ],
            vec![translating.clone(), entry],
        )
        .unwrap();

        let segment = indices
            .iter()
            .find(|index| index.uuid == translating.uuid)
            .unwrap();
        assert!(!segment.fragment_bitmap.as_ref().unwrap().contains(7));
    }

    /// The legacy overlay operation is refused on a table with a tagged
    /// history, and so is its action.
    #[test]
    fn test_an_overlay_is_refused_beside_a_tagged_history() {
        let error = apply_with_indices(
            &table(3, false),
            vec![Action::AddOverlays(AddOverlays {
                fragment: Ref::Committed(0),
                overlays: vec![DataOverlayFile {
                    data_file: DataFile::new_legacy_from_fields("o.lance", vec![0], None),
                    coverage: OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([0u32])),
                    committed_version: 1,
                }],
                data_change: true,
            })],
            vec![history(1, Vec::new(), vec![recorded(&[7], &[0])])],
        )
        .unwrap_err();

        assert!(matches!(error, Error::NotSupported { .. }), "{error:?}");
    }

    #[test]
    fn test_round_trips_through_the_wire() {
        let decoded =
            TrimFragReuse::try_from(pb_actions::TrimFragReuse::from(&TrimFragReuse {})).unwrap();
        assert_eq!(decoded, TrimFragReuse {});
    }
}
