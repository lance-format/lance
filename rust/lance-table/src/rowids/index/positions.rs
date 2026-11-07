// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{EncodedU64Array, Result, batch};
use lance_core::deepsize::DeepSizeOf;

/// An unsorted segment's inverse permutation, without retaining its source IDs.
/// Dense keys need only a membership bitmap, rank directory and physical offsets.
/// Sparse keys use packed (relative ID, offset) words when the ID span permits it.
#[derive(Debug, DeepSizeOf)]
pub(super) enum Positions {
    Dense {
        base: u64,
        words: Vec<u64>,
        ranks: Vec<u32>,
        offsets: Vec<u32>,
    },
    Packed {
        base: u64,
        entries: Vec<u64>,
    },
    Wide {
        entries: Vec<WidePosition>,
    },
}

/// High word first preserves numeric ID order without padding each entry to
/// 16 bytes. Sorting direct keys avoids random reads from the source array.
#[derive(Debug, DeepSizeOf, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct WidePosition {
    key: [u32; 2],
    offset: u32,
}

impl Positions {
    pub(super) fn new(values: &EncodedU64Array) -> Self {
        let Some(base) = values.min() else {
            return Self::Packed {
                base: 0,
                entries: Vec::new(),
            };
        };
        let max = values.max().unwrap_or(base);
        // At density >= 1/16 the bitmap and word ranks cost at most 3 bytes
        // per row, less than a packed key. Small arrays avoid directory overhead.
        let dense_span = max
            .checked_sub(base)
            .and_then(|span| span.checked_add(1))
            .filter(|&span| values.len() >= 64 && span <= values.len() as u64 * 16)
            .and_then(|span| usize::try_from(span).ok());
        if let Some(span) = dense_span {
            let mut words = vec![0_u64; span.div_ceil(64)];
            for id in values.iter() {
                let relative = (id - base) as usize;
                words[relative / 64] |= 1 << (relative % 64);
            }
            let mut ranks = Vec::with_capacity(words.len());
            let mut count = 0_u32;
            for word in &words {
                ranks.push(count);
                count += word.count_ones();
            }
            let mut offsets = vec![u32::MAX; count as usize];
            for (offset, id) in values.iter().enumerate() {
                let relative = (id - base) as usize;
                let word = relative / 64;
                let lower = (1_u64 << (relative % 64)) - 1;
                let rank = ranks[word] + (words[word] & lower).count_ones();
                // Match the source segment's first-position semantics for duplicates.
                offsets[rank as usize] = offsets[rank as usize].min(offset as u32);
            }
            Self::Dense {
                base,
                words,
                ranks,
                offsets,
            }
        } else if u32::try_from(max - base).is_ok() {
            let mut entries: Vec<u64> = values
                .iter()
                .enumerate()
                .map(|(offset, id)| ((id - base) << 32) | offset as u64)
                .collect();
            entries.sort_unstable();
            entries.dedup_by_key(|entry| *entry >> 32);
            Self::Packed { base, entries }
        } else {
            let mut entries: Vec<_> = values
                .iter()
                .enumerate()
                .map(|(offset, id)| WidePosition {
                    key: [(id >> 32) as u32, id as u32],
                    offset: offset as u32,
                })
                .collect();
            entries.sort_unstable();
            entries.dedup_by_key(|entry| entry.key);
            Self::Wide { entries }
        }
    }

    pub(super) fn get(&self, id: u64) -> Option<u32> {
        match self {
            Self::Dense {
                base,
                words,
                ranks,
                offsets,
            } => {
                let relative = usize::try_from(id.checked_sub(*base)?).ok()?;
                let word = relative / 64;
                let bits = *words.get(word)?;
                let bit = 1_u64 << (relative % 64);
                if bits & bit == 0 {
                    return None;
                }
                let rank = ranks[word] + (bits & (bit - 1)).count_ones();
                Some(offsets[rank as usize])
            }
            Self::Packed { base, entries } => {
                let relative = id.checked_sub(*base)?;
                if relative > u32::MAX as u64 {
                    return None;
                }
                entries
                    .binary_search_by_key(&relative, |entry| entry >> 32)
                    .ok()
                    .map(|found| entries[found] as u32)
            }
            Self::Wide { entries } => entries
                .binary_search_by_key(&[(id >> 32) as u32, id as u32], |entry| entry.key)
                .ok()
                .map(|found| entries[found].offset),
        }
    }

    pub(super) fn search_many(
        &self,
        queries: &[(u64, usize)],
        emit: &mut impl FnMut(u64, usize, u32) -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Dense { .. } => {
                for &(id, ordinal) in queries {
                    if let Some(offset) = self.get(id) {
                        emit(id, ordinal, offset)?;
                    }
                }
                Ok(())
            }
            Self::Packed { base, entries } => batch::search_sorted(
                entries.len(),
                |i| *base + (entries[i] >> 32),
                |i| entries[i] as u32,
                queries,
                emit,
            ),
            Self::Wide { entries } => batch::search_sorted(
                entries.len(),
                |i| ((entries[i].key[0] as u64) << 32) | entries[i].key[1] as u64,
                |i| entries[i].offset,
                queries,
                emit,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::dense(1, 0)]
    #[case::packed(1000, 1 << 40)]
    #[case::wide(1 << 40, 0)]
    #[case::wide_carry(u32::MAX as u64, (1 << 32) - 1)]
    fn inverse_matches_physical_order(#[case] stride: u64, #[case] base: u64) {
        let mut ids: Vec<_> = (0..257).rev().map(|i| base + i * stride).collect();
        ids.extend_from_within(0..5);
        let positions = Positions::new(&ids.clone().into());
        let queries: Vec<_> = ids
            .iter()
            .copied()
            .chain([0, u64::MAX, base + 1])
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();
        let mut actual = vec![None; queries.len()];
        positions
            .search_many(&queries, &mut |_, i, offset| {
                actual[i] = Some(offset);
                Ok(())
            })
            .unwrap();
        for &(id, i) in &queries {
            let expected = ids
                .iter()
                .position(|&candidate| candidate == id)
                .map(|i| i as u32);
            assert_eq!(positions.get(id), expected);
            assert_eq!(actual[i], expected);
        }
    }

    #[test]
    fn full_width_bounds_and_empty() {
        for ids in [
            vec![],
            vec![u64::MAX, 0, u64::MAX],
            vec![u64::MAX; 128],
            (0..128).rev().map(|i| u64::MAX - i).collect(),
        ] {
            let positions = Positions::new(&ids.clone().into());
            for id in ids.iter().copied().chain([0, 1, u64::MAX - 128, u64::MAX]) {
                assert_eq!(
                    positions.get(id),
                    ids.iter().position(|&v| v == id).map(|i| i as u32)
                );
            }
        }
    }
}
