// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{Error, Result, RowAddress, RowIdIndex, SegmentLookup};

/// Independent searches to advance together, hiding dependent-load latency
/// without the register pressure and long scalar tails of wider batches.
pub(super) const LANES: usize = 32;

impl RowIdIndex {
    pub(super) fn probe_many(
        &self,
        queries: &[(u64, usize)],
        out: &mut [Option<RowAddress>],
    ) -> Result<()> {
        self.probe_batch_node(1, 0, self.end_tree.len() / 2, queries, out)
    }

    fn probe_batch_node(
        &self,
        node: usize,
        lo: usize,
        width: usize,
        queries: &[(u64, usize)],
        out: &mut [Option<RowAddress>],
    ) -> Result<()> {
        let Some(probe) = self.probes.get(lo) else {
            return Ok(());
        };
        // All leaves in this subtree start at or after `probe.start`, and none
        // ends after `end_tree[node]`. Prune for the batch, not the whole index:
        // a few queries among many disjoint pieces must not scan every probe.
        let first = queries.partition_point(|&(id, _)| id < probe.start);
        let last = queries.partition_point(|&(id, _)| id <= self.end_tree[node]);
        let queries = &queries[first..last];
        if queries.is_empty() {
            return Ok(());
        }
        if width > 1 {
            self.probe_batch_node(node * 2, lo, width / 2, queries, out)?;
            return self.probe_batch_node(node * 2 + 1, lo + width / 2, width / 2, queries, out);
        }

        let fragment = &self.fragments[probe.fragment as usize];
        let entry = &fragment.segments[probe.segment as usize];
        let mut store = |id, ordinal: usize, address| {
            // Keep probing after a hit: another live copy is corruption.
            if out[ordinal].is_some() {
                return Err(Error::internal(format!(
                    "row id index corrupt: stable row id {id} is live in multiple fragments"
                )));
            }
            out[ordinal] = Some(address);
            Ok(())
        };
        if let SegmentLookup::Positions(positions) = &entry.lookup {
            positions.search_many(queries, &mut |id, ordinal, position| {
                let offset = entry.start_offset + position;
                if !fragment.deletion_vector.contains(offset) {
                    store(
                        id,
                        ordinal,
                        RowAddress::new_from_parts(fragment.fragment_id, offset),
                    )?;
                }
                Ok(())
            })
        } else {
            for &(id, ordinal) in queries {
                if let Some(address) = fragment.resolve_segment(probe.segment as usize, id) {
                    store(id, ordinal, address)?;
                }
            }
            Ok(())
        }
    }
}

pub(super) fn search_sorted(
    len: usize,
    key: impl Fn(usize) -> u64,
    offset: impl Fn(usize) -> u32,
    queries: &[(u64, usize)],
    emit: &mut impl FnMut(u64, usize, u32) -> Result<()>,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    let (chunks, remainder) = queries.as_chunks::<LANES>();
    for chunk in chunks {
        let mut bases = [0usize; LANES];
        let mut size = len;
        while size > 1 {
            let half = size / 2;
            let mids: [usize; LANES] = std::array::from_fn(|lane| bases[lane] + half);
            // Load independent midpoints before consuming their results. Each
            // lane still depends on its own previous comparison, not its peers.
            let keys: [u64; LANES] = std::array::from_fn(|lane| key(mids[lane]));
            for lane in 0..LANES {
                bases[lane] = std::hint::select_unpredictable(
                    keys[lane] <= chunk[lane].0,
                    mids[lane],
                    bases[lane],
                );
            }
            size -= half;
        }
        for lane in 0..LANES {
            let (id, ordinal) = chunk[lane];
            if key(bases[lane]) == id {
                emit(id, ordinal, offset(bases[lane]))?;
            }
        }
    }
    for &(id, ordinal) in remainder {
        let mut base = 0;
        let mut size = len;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            base = std::hint::select_unpredictable(key(mid) <= id, mid, base);
            size -= half;
        }
        if key(base) == id {
            emit(id, ordinal, offset(base))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rowids::{FragmentRowIdIndex, RowIdSequence, U64Segment};
    use lance_core::utils::deletion::DeletionVector;
    use proptest::prelude::*;
    use std::sync::Arc;

    fn check_search(mut keys: Vec<u64>, queries: Vec<u64>) {
        keys.sort_unstable();
        keys.dedup();
        // Physical positions intentionally differ from sorted ranks.
        let positions: Vec<_> = keys
            .iter()
            .enumerate()
            .map(|(i, &key)| (key, (keys.len() - i) as u32))
            .collect();
        let expected: Vec<_> = queries
            .iter()
            .map(|id| positions.iter().find(|&&(key, _)| key == *id).map(|p| p.1))
            .collect();
        let mut sorted: Vec<_> = queries.into_iter().zip(0..expected.len()).collect();
        sorted.sort_unstable_by_key(|q| q.0);
        let mut actual = vec![None; expected.len()];
        search_sorted(
            positions.len(),
            |i| positions[i].0,
            |i| positions[i].1,
            &sorted,
            &mut |_, ordinal, offset| {
                assert!(actual[ordinal].is_none());
                actual[ordinal] = Some(offset);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[rstest::rstest]
    #[case::empty(0)]
    #[case::singleton(1)]
    #[case::two(2)]
    #[case::before_batch(31)]
    #[case::full_batch(32)]
    #[case::after_batch(33)]
    #[case::before_two_batches(63)]
    #[case::two_batches(64)]
    #[case::after_two_batches(65)]
    #[case::uneven_large(257)]
    fn boundaries_and_tails(#[case] len: usize) {
        let keys: Vec<_> = (0..len as u64).map(|i| i * 2).chain([u64::MAX]).collect();
        for count in [0, 1, 31, 32, 33, 63, 64, 65, 1000] {
            let queries = (0..count)
                .rev()
                .map(|i| if i % 3 == 0 { u64::MAX } else { i % 100 })
                .collect::<Vec<_>>();
            check_search(keys.clone(), queries.clone());
            check_search(vec![], queries);
        }
    }

    proptest! {
        #[test]
        fn search_matches_linear_oracle(
            keys in prop::collection::vec(any::<u64>(), 0..128),
            queries in prop::collection::vec(any::<u64>(), 0..260),
        ) {
            let queries = queries.into_iter().chain(keys.iter().copied()).collect();
            check_search(keys, queries);
        }

        #[test]
        fn tree_matches_physical_rows(
            fragment_count in 1usize..10,
            rows in prop::collection::vec((0u16..1024, any::<bool>()), 0..128),
            queries in prop::collection::vec(0u16..1100, 0..260),
        ) {
            let mut rows = rows;
            rows.sort_unstable_by_key(|r| r.0);
            rows.dedup_by_key(|r| r.0);
            let mut fragments = Vec::new();
            let mut live = std::collections::BTreeMap::new();
            for f in 0..fragment_count {
                let selected: Vec<_> = rows.iter().rev()
                    .filter(|r| usize::from(r.0) % fragment_count == f).collect();
                let ids: Vec<_> = selected.iter().map(|r| u64::from(r.0) * (1 << 17)).collect();
                let deleted: Vec<_> = selected.iter().enumerate()
                    .filter(|(_, r)| r.1).map(|(i, _)| i as u32).collect();
                for (offset, r) in selected.iter().enumerate().filter(|(_, r)| !r.1) {
                    live.insert(u64::from(r.0) * (1 << 17), RowAddress::new_from_parts(f as u32, offset as u32));
                }
                fragments.push(fragment(f as u32, vec![U64Segment::Array(ids.into())], &deleted));
            }
            let queries: Vec<_> = queries.into_iter().map(|id| u64::from(id) * (1 << 17))
                .chain(live.keys().copied()).chain([u64::MAX]).collect();
            let expected: Vec<_> = queries.iter().map(|id| live.get(id).copied()).collect();
            let index = RowIdIndex::probing(&fragments).unwrap();
            prop_assert_eq!(index.get_many(&queries).unwrap(), expected);
        }
    }

    fn fragment(id: u32, segments: Vec<U64Segment>, deleted: &[u32]) -> FragmentRowIdIndex {
        FragmentRowIdIndex {
            fragment_id: id,
            row_id_sequence: Arc::new(RowIdSequence(segments)),
            deletion_vector: Arc::new(DeletionVector::Bitmap(deleted.iter().copied().collect())),
        }
    }

    #[test]
    fn mixed_segments_offsets_deletions_and_merged_fallback() {
        let far = 1 << 20;
        let fragments = [
            fragment(
                3,
                vec![
                    U64Segment::Range(10..13),
                    U64Segment::Array(vec![50, 20, 40, 30].into()),
                    U64Segment::SortedArray(vec![100, 102, far].into()),
                ],
                &[4],
            ),
            fragment(
                9,
                vec![
                    U64Segment::Array(vec![80, 20, 60].into()),
                    U64Segment::RangeWithBitmap {
                        range: 200..205,
                        bitmap: [true, false, true, true, false].as_slice().into(),
                    },
                    U64Segment::RangeWithHoles {
                        range: 300..304,
                        holes: vec![301].into(),
                    },
                ],
                &[2],
            ),
        ];
        // Explicit physical offsets, independent of the search implementation.
        let live = [
            (10, 3, 0),
            (11, 3, 1),
            (12, 3, 2),
            (50, 3, 3),
            (40, 3, 5),
            (30, 3, 6),
            (100, 3, 7),
            (102, 3, 8),
            (far, 3, 9),
            (80, 9, 0),
            (20, 9, 1),
            (200, 9, 3),
            (202, 9, 4),
            (203, 9, 5),
            (300, 9, 6),
            (302, 9, 7),
            (303, 9, 8),
        ];
        let queries: Vec<_> = (0..320)
            .rev()
            .chain([far, far + 1, 20, 20, 10, u64::MAX])
            .chain(std::iter::repeat_n(40, 256))
            .collect();
        let expected: Vec<_> = queries
            .iter()
            .map(|q| {
                live.iter()
                    .find(|p| p.0 == *q)
                    .map(|p| RowAddress::new_from_parts(p.1, p.2))
            })
            .collect();
        for index in [
            RowIdIndex::probing(&fragments).unwrap(),
            RowIdIndex::new(&fragments).unwrap(),
        ] {
            assert_eq!(index.get_many(&queries).unwrap(), expected);
            assert!(index.get_many(&[]).unwrap().is_empty());
            assert_eq!(index.get_many(&[999, u64::MAX]).unwrap(), vec![None, None]);
        }
    }

    #[test]
    fn conflicting_live_ids_are_errors() {
        let index = RowIdIndex::probing(&[
            fragment(1, vec![U64Segment::Array(vec![8, 2].into())], &[]),
            fragment(2, vec![U64Segment::Array(vec![7, 2].into())], &[]),
        ])
        .unwrap();
        for queries in [vec![8, 2, 2], (0..384).map(|i| [8, 7, 2][i % 3]).collect()] {
            let error = index.get_many(&queries).unwrap_err();
            assert!(matches!(error, Error::Internal { .. }));
            assert!(
                error
                    .to_string()
                    .contains("stable row id 2 is live in multiple fragments")
            );
        }
    }

    #[test]
    fn empty_tree_boundaries_and_repeated_segment_ids() {
        let queries: Vec<_> = (0..65).map(|i| [8, u64::MAX, 0, 1][i % 4]).collect();
        assert_eq!(
            RowIdIndex::probing(&[])
                .unwrap()
                .get_many(&queries)
                .unwrap(),
            vec![None; 65]
        );
        let fragments = [fragment(
            5,
            vec![U64Segment::Array(vec![8, 0, 8, u64::MAX].into())],
            &[],
        )];
        assert_eq!(
            RowIdIndex::probing(&fragments)
                .unwrap()
                .get_many(&queries)
                .unwrap(),
            queries
                .iter()
                .map(|id| match *id {
                    8 => Some(RowAddress::new_from_parts(5, 0)),
                    0 => Some(RowAddress::new_from_parts(5, 1)),
                    u64::MAX => Some(RowAddress::new_from_parts(5, 3)),
                    _ => None,
                })
                .collect::<Vec<_>>()
        );
    }
}
