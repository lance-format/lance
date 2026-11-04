// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Composing a version 0 history into the rewrite of one segment: whether
//! the segment is rewritten and the bitmap it publishes, decided as the
//! released version 0 remap always decided them.

use std::sync::Arc;

use lance_core::{Error, Result};

use super::super::FriImpl;
use crate::system_index::frag_reuse::CompactFragReuseIndex;
use crate::system_index::frag_reuse::plan::{Composition, SegmentSummary, TranslationPlan};

/// Every compaction version is applied to the segment's stored bitmap in
/// one pass, in dataset version order, so a version sees the fragments an
/// earlier one added as its sources. A group whose sources the bitmap holds
/// is swapped to its destinations, and one it holds part of is refused. A
/// group whose destinations the bitmap already holds is still rewritten
/// when the segment was built at or before the version (the comparison is
/// inclusive): a commit advanced the bitmap before the segment's data was
/// remapped, for instance while remapping a sibling. A segment without a
/// bitmap is always rewritten.
///
/// The rewrite moves addresses through the whole history, so the plan's
/// coverage does not limit what it translates.
pub(super) fn compose(
    history: Arc<dyn FriImpl>,
    index: &CompactFragReuseIndex,
    segment: &SegmentSummary,
) -> Result<Composition> {
    if index.is_empty() {
        return Ok(Composition::CaughtUp);
    }
    let Some(stored) = &segment.fragment_bitmap else {
        return Ok(Composition::Rewrite(TranslationPlan::legacy_rewrite(
            history, None,
        )));
    };
    let mut bitmap = stored.clone();
    let mut is_stale = false;
    for version in &index.details.versions {
        let data_predates_version = segment.dataset_version <= version.dataset_version;
        for group in &version.groups {
            let mut old_frag_in_index = 0;
            for old_frag in &group.old_frags {
                if bitmap.remove(old_frag.id as u32) {
                    old_frag_in_index += 1;
                }
            }
            if old_frag_in_index > 0 {
                if old_frag_in_index != group.old_frags.len() {
                    // This should never happen because we always commit a full rewrite group
                    // and we always reindex either the entire group or nothing.
                    // We use invalid input to be consistent with
                    // dataset::transaction::recalculate_fragment_bitmap
                    return Err(Error::invalid_input(format!(
                        "The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {:?}",
                        group.old_frags
                    )));
                }
                bitmap.extend(group.new_frags.iter().map(|frag| frag.id as u32));
                is_stale = true;
            } else if data_predates_version
                && group
                    .new_frags
                    .iter()
                    .any(|new_frag| bitmap.contains(new_frag.id as u32))
            {
                is_stale = true;
            }
        }
    }
    Ok(if is_stale {
        Composition::Rewrite(TranslationPlan::legacy_rewrite(history, Some(bitmap)))
    } else {
        Composition::CaughtUp
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use lance_core::Error;
    use roaring::RoaringBitmap;

    use crate::system_index::frag_reuse::history::FragReuseHistory;
    use crate::system_index::frag_reuse::history::tests::{address, legacy, whole};
    use crate::system_index::frag_reuse::mapping::MappingSource;
    use crate::system_index::frag_reuse::plan::{
        Composition, RewriteContext, RowDomain, SegmentSummary, TranslationPlan, TranslationSupport,
    };

    /// Fragments 0 and 1 into 2 at version 10, then 2 and 3 into 4 at 20.
    fn two_rounds() -> FragReuseHistory {
        FragReuseHistory::from_legacy_versions(vec![
            legacy(10, &[whole(0, 2), whole(1, 2)], &[(2, 4)]),
            legacy(20, &[whole(2, 4), whole(3, 2)], &[(4, 6)]),
        ])
        .unwrap()
    }

    fn segment(covered: Option<&[u32]>, dataset_version: u64) -> SegmentSummary {
        SegmentSummary {
            uuid: uuid::Uuid::new_v4(),
            fragment_bitmap: covered.map(|covered| covered.iter().copied().collect()),
            dataset_version,
            domain: RowDomain::Address,
            translation: TranslationSupport::Supported,
        }
    }

    fn compose(history: &FragReuseHistory, segment: &SegmentSummary) -> Composition {
        let live = RoaringBitmap::from_iter([4]);
        let lineage = history.fragment_bitmap();
        let rows = HashMap::new();
        history
            .compose(
                &RewriteContext {
                    live_fragments: &live,
                    lineage: &lineage,
                    fragment_rows: &rows,
                },
                segment,
            )
            .unwrap()
    }

    fn rewritten(composition: Composition) -> TranslationPlan {
        match composition {
            Composition::Rewrite(plan) => plan,
            other => panic!("expected a rewrite, got {other:?}"),
        }
    }

    #[rstest::rstest]
    #[case::sources_chain_through_both_rounds(&[0, 1, 3, 9], 5, &[4, 9])]
    #[case::second_round_only(&[2, 3], 15, &[4])]
    // A sibling's remap commit advanced the bitmap onto the destinations
    // before this segment's data was remapped: the bitmap stays, the data is
    // still rewritten, up to and including the version's own dataset version.
    #[case::advanced_bitmap_predates_the_version(&[4], 20, &[4])]
    #[case::advanced_bitmap_through_an_earlier_round(&[2, 3], 10, &[4])]
    fn stale_segments_are_rewritten(
        #[case] covered: &[u32],
        #[case] dataset_version: u64,
        #[case] published: &[u32],
    ) {
        let plan = rewritten(compose(
            &two_rounds(),
            &segment(Some(covered), dataset_version),
        ));
        assert_eq!(
            plan.published_bitmap(),
            Some(&published.iter().copied().collect())
        );
        assert!(plan.moves_addresses());
    }

    #[rstest::rstest]
    #[case::disjoint(&[7, 8], 1)]
    #[case::built_after_the_last_round(&[4], 21)]
    #[case::empty_bitmap(&[], 1)]
    fn caught_up_segments_are_left_alone(#[case] covered: &[u32], #[case] dataset_version: u64) {
        assert_eq!(
            compose(&two_rounds(), &segment(Some(covered), dataset_version)),
            Composition::CaughtUp
        );
    }

    /// Released writers did not always record a bitmap; such a segment is
    /// rewritten whatever its version, and publishes no bitmap.
    #[rstest::rstest]
    #[case::before(1)]
    #[case::after(21)]
    fn a_segment_without_a_bitmap_is_rewritten_without_one(#[case] dataset_version: u64) {
        let plan = rewritten(compose(&two_rounds(), &segment(None, dataset_version)));
        assert_eq!(plan.published_bitmap(), None);
        assert!(plan.moves_addresses());
    }

    #[test]
    fn an_empty_history_rewrites_nothing() {
        let history = FragReuseHistory::from_legacy_versions(vec![]).unwrap();
        assert_eq!(compose(&history, &segment(None, 1)), Composition::CaughtUp);
        assert_eq!(
            compose(&history, &segment(Some(&[0]), 1)),
            Composition::CaughtUp
        );
    }

    /// Only a corrupted version records a group a segment covers part of.
    #[rstest::rstest]
    #[case::before(1)]
    #[case::after(21)]
    fn a_partly_covered_group_is_refused(#[case] dataset_version: u64) {
        let history = two_rounds();
        let live = RoaringBitmap::new();
        let lineage = history.fragment_bitmap();
        let rows = HashMap::new();
        let error = history
            .compose(
                &RewriteContext {
                    live_fragments: &live,
                    lineage: &lineage,
                    fragment_rows: &rows,
                },
                &segment(Some(&[0]), dataset_version),
            )
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("a rewrite group that was a split of indexed and non-indexed data"),
            "{error}"
        );
    }

    #[derive(Debug)]
    struct NoMappings;

    #[async_trait::async_trait]
    impl MappingSource for NoMappings {
        async fn counts(
            &self,
            _: &crate::system_index::frag_reuse::mapping::MappingRef,
        ) -> lance_core::Result<Arc<lance_core::utils::stable_partition::CountsMatrix>> {
            unreachable!("a version 0 history has no external mappings")
        }

        async fn read(
            &self,
            _: &crate::system_index::frag_reuse::mapping::MappingRef,
            _: std::ops::Range<u64>,
        ) -> lance_core::Result<arrow_array::UInt16Array> {
            unreachable!("a version 0 history has no external mappings")
        }
    }

    /// The rewrite translates through every version, as one in-memory map,
    /// whatever the published bitmap claims: an address outside the history
    /// stays as it is.
    #[tokio::test]
    async fn the_rewrite_translates_through_the_whole_history() {
        let history = two_rounds();
        let plan = rewritten(compose(&history, &segment(Some(&[2, 3]), 15)));
        let remapper = plan.bind(Arc::new(NoMappings));
        // An index rewrite applies it as that one map.
        assert!(remapper.row_addr_remap().is_some());
        let mut addrs = vec![
            Some(address(0, 1)),
            Some(address(3, 0)),
            Some(address(9, 0)),
            None,
        ];
        remapper.translate_in_place(&mut addrs).await.unwrap();
        assert_eq!(
            addrs,
            vec![
                Some(address(4, 1)),
                Some(address(4, 4)),
                Some(address(9, 0)),
                None
            ]
        );
    }
}
