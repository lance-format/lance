// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compose sorted prepared maps without materializing all key/address pairs.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::RangeInclusive;
use std::sync::Arc;

use lance_core::{Error, Result};

use super::{FragmentIdentity, MergedChunk, RowIdIndex, U64Segment};

/// At most 16 MiB of decoded IDs and addresses per merge. Encoded output
/// chunks remain separately searchable by the existing merged lookup path.
const OUTPUT_ROWS: usize = 1 << 20;

type Chunk = (RangeInclusive<u64>, Arc<MergedChunk>);

pub(super) fn merged(
    indices: &[&RowIdIndex],
    sources: Vec<FragmentIdentity>,
) -> Result<RowIdIndex> {
    let mut chunks: Vec<_> = indices
        .iter()
        .filter_map(|index| index.merged.as_ref())
        .flat_map(|index| index.iter())
        .collect();
    chunks.sort_unstable_by_key(|(range, _)| *range.start());
    let group = Arc::new(sources.clone());
    let mut output = Vec::with_capacity(chunks.len());
    let mut start = 0;
    while start < chunks.len() {
        let mut end = start + 1;
        let mut range_end = *chunks[start].0.end();
        while end < chunks.len() && *chunks[end].0.start() <= range_end {
            range_end = range_end.max(*chunks[end].0.end());
            end += 1;
        }
        if end == start + 1 {
            // Sharing the chunk preserves its original source group and never
            // retains the input RowIdIndex wrapper or its other chunks.
            output.push((chunks[start].0.clone(), chunks[start].1.clone()));
        } else {
            merge_component(&chunks[start..end], &group, &mut output, OUTPUT_ROWS)?;
        }
        start = end;
    }
    Ok(RowIdIndex {
        fragments: Vec::new(),
        probes: Vec::new(),
        end_tree: Vec::new(),
        merged: Some(output.into_iter().collect()),
        sources,
    })
}

fn merge_component(
    chunks: &[(&RangeInclusive<u64>, &Arc<MergedChunk>)],
    sources: &Arc<Vec<FragmentIdentity>>,
    output: &mut Vec<Chunk>,
    output_rows: usize,
) -> Result<()> {
    debug_assert!(
        output_rows > 0,
        "merged output chunks must hold at least one row"
    );
    let mut remaining = chunks.iter().try_fold(0usize, |total, (_, chunk)| {
        total.checked_add(chunk.segments.0.len()).ok_or_else(|| {
            Error::invalid_input("prepared merged row count exceeds addressable memory")
        })
    })?;
    let mut cursors: Vec<_> = chunks
        .iter()
        .map(|(_, chunk)| chunk.segments.0.iter().zip(chunk.segments.1.iter()))
        .collect();
    let mut heap = BinaryHeap::with_capacity(cursors.len());
    for (input, cursor) in cursors.iter_mut().enumerate() {
        if let Some((id, address)) = cursor.next() {
            heap.push(Reverse((id, address, input)));
        }
    }
    let mut ids = Vec::with_capacity(remaining.min(output_rows));
    let mut addresses = Vec::with_capacity(remaining.min(output_rows));
    let mut previous = None;
    while let Some(mut first) = heap.peek_mut() {
        let Reverse((id, address, input)) = *first;
        if previous == Some(id) {
            return Err(Error::internal(format!(
                "row id index corrupt: stable row id {id} is live in multiple fragments"
            )));
        }
        previous = Some(id);
        ids.push(id);
        addresses.push(address);
        remaining -= 1;
        if ids.len() == output_rows {
            append_chunk(
                std::mem::take(&mut ids),
                std::mem::take(&mut addresses),
                sources,
                output,
            );
            ids = Vec::with_capacity(remaining.min(output_rows));
            addresses = Vec::with_capacity(remaining.min(output_rows));
        }
        if let Some((id, address)) = cursors[input].next() {
            *first = Reverse((id, address, input));
        } else {
            std::collections::binary_heap::PeekMut::pop(first);
        }
    }
    append_chunk(ids, addresses, sources, output);
    Ok(())
}

fn append_chunk(
    ids: Vec<u64>,
    addresses: Vec<u64>,
    sources: &Arc<Vec<FragmentIdentity>>,
    output: &mut Vec<Chunk>,
) {
    debug_assert_eq!(ids.len(), addresses.len(), "paired prepared map lengths");
    if let Some((&start, &end)) = ids.first().zip(ids.last()) {
        output.push((
            start..=end,
            Arc::new(MergedChunk {
                segments: (U64Segment::from_iter(ids), U64Segment::from_iter(addresses)),
                sources: sources.clone(),
            }),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rowids::{FragmentRowIdIndex, RowIdSequence};
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::deletion::DeletionVector;

    #[test]
    fn merged_composition_preserves_inclusive_endpoints_with_empty_sources() {
        let sources: Vec<_> = [vec![u64::MAX, 0], vec![u64::MAX - 1, 1], vec![]]
            .into_iter()
            .enumerate()
            .map(|(fragment_id, ids)| FragmentRowIdIndex {
                fragment_id: fragment_id as u32,
                row_id_sequence: Arc::new(RowIdSequence::try_from_iter(ids).unwrap()),
                deletion_vector: Arc::new(DeletionVector::default()),
            })
            .collect();
        let indices: Vec<_> = sources
            .iter()
            .map(|source| RowIdIndex::new(std::slice::from_ref(source)).unwrap())
            .collect();
        assert!(indices.iter().all(|index| index.merged.is_some()));
        let combined = RowIdIndex::try_combine(&[&indices[2], &indices[0], &indices[1]])
            .unwrap()
            .unwrap();
        let ids = [u64::MAX, 0, u64::MAX - 1, 1, 2, u64::MAX - 2, u64::MAX];
        let expected = [
            Some(RowAddress::new_from_parts(0, 0)),
            Some(RowAddress::new_from_parts(0, 1)),
            Some(RowAddress::new_from_parts(1, 0)),
            Some(RowAddress::new_from_parts(1, 1)),
            None,
            None,
            Some(RowAddress::new_from_parts(0, 0)),
        ];
        assert_eq!(combined.get_many(&ids).unwrap(), expected);
        for (id, address) in ids.into_iter().zip(expected) {
            assert_eq!(combined.get(id).unwrap(), address);
        }
    }

    #[test]
    fn bounded_chunks_preserve_pairs_and_duplicate_validation_across_flushes() {
        let sources: Vec<_> = (0..2)
            .map(|fragment_id| FragmentRowIdIndex {
                fragment_id,
                row_id_sequence: Arc::new(
                    RowIdSequence::try_from_iter([
                        u64::from(fragment_id) + 4,
                        u64::from(fragment_id),
                        u64::from(fragment_id) + 2,
                    ])
                    .unwrap(),
                ),
                deletion_vector: Arc::new(DeletionVector::default()),
            })
            .collect();
        let indices: Vec<_> = sources
            .iter()
            .map(|source| RowIdIndex::new(std::slice::from_ref(source)).unwrap())
            .collect();
        let chunks: Vec<_> = indices
            .iter()
            .flat_map(|index| index.merged.as_ref().unwrap().iter())
            .collect();
        let identities = Arc::new(sources.iter().map(FragmentIdentity::new).collect());
        let mut output = Vec::new();
        merge_component(&chunks, &identities, &mut output, 2).unwrap();
        assert_eq!(output.len(), 3);
        for (part, (range, chunk)) in output.iter().enumerate() {
            assert_eq!(range, &(part as u64 * 2..=part as u64 * 2 + 1));
            assert_eq!(chunk.segments.0.len(), 2);
            for (id, address) in chunk.segments.0.iter().zip(chunk.segments.1.iter()) {
                let offset = [1, 2, 0][id as usize / 2];
                assert_eq!(
                    RowAddress::from(address),
                    RowAddress::new_from_parts(id as u32 % 2, offset)
                );
            }
        }
        // With a three-row buffer the duplicate 2 crosses the flush boundary.
        let other = RowIdIndex::new(&[FragmentRowIdIndex {
            fragment_id: 9,
            row_id_sequence: Arc::new(RowIdSequence::try_from_iter([2, 8]).unwrap()),
            deletion_vector: Arc::new(DeletionVector::default()),
        }])
        .unwrap();
        let mut duplicated = chunks;
        duplicated.extend(other.merged.as_ref().unwrap().iter());
        let error = merge_component(&duplicated, &identities, &mut Vec::new(), 3).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(
            error
                .to_string()
                .contains("stable row id 2 is live in multiple fragments")
        );
    }
}
