// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use lance_core::{Error, Result};

/// Number of upper-triangular comparison blocks for `rows` storage rows.
/// Off-diagonal blocks compare all pairs between two ranges; diagonal blocks
/// compare unique pairs within one range. Each range has at most `rows_per_block` rows.
///
/// ```
/// use lance::index::vector::dedup::pairwise_block_count;
/// assert_eq!(pairwise_block_count(300_000, 100_000).unwrap(), 6);
/// ```
pub fn pairwise_block_count(rows: usize, rows_per_block: usize) -> Result<usize> {
    if rows_per_block == 0 {
        return Err(Error::invalid_input("rows_per_block must be positive"));
    }
    let axes = rows.div_ceil(rows_per_block) as u128;
    usize::try_from(axes * (axes + 1) / 2)
        .map_err(|_| Error::invalid_input("comparison block count exceeds usize"))
}

pub(super) fn assigned_blocks(
    rows: usize,
    rows_per_block: usize,
    worker_index: usize,
    worker_count: usize,
) -> Result<impl Iterator<Item = (Range<usize>, Range<usize>)>> {
    if worker_count == 0 || worker_index >= worker_count {
        return Err(Error::invalid_input(
            "invalid comparison block worker assignment",
        ));
    }
    let count = pairwise_block_count(rows, rows_per_block)?;
    let axes = rows.div_ceil(rows_per_block);
    let prefix = move |axis: usize| {
        let axis = axis as u128;
        axis * (2 * axes as u128 - axis + 1) / 2
    };
    Ok((worker_index..count)
        .step_by(worker_count)
        .map(move |ordinal| {
            let mut low = 0;
            let mut high = axes;
            while low < high {
                let middle = low + (high - low) / 2;
                if prefix(middle) <= ordinal as u128 {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            let anchor = low - 1;
            let candidate = anchor + (ordinal as u128 - prefix(anchor)) as usize;
            let range = |axis: usize| {
                let start = axis * rows_per_block;
                start..start.saturating_add(rows_per_block).min(rows)
            };
            (range(anchor), range(candidate))
        }))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{assigned_blocks, pairwise_block_count};

    #[test]
    fn comparison_blocks_cover_each_pair_once() {
        for rows in 0..20 {
            for block_rows in [1, 3, 8, 100_000] {
                for workers in [1, 2, 4, 16] {
                    let mut pairs = BTreeSet::new();
                    let mut blocks = 0;
                    for worker in 0..workers {
                        for (left, right) in
                            assigned_blocks(rows, block_rows, worker, workers).unwrap()
                        {
                            blocks += 1;
                            for a in left.clone() {
                                for b in right.clone() {
                                    if a < b {
                                        assert!(pairs.insert((a, b)));
                                    }
                                }
                            }
                        }
                    }
                    assert_eq!(blocks, pairwise_block_count(rows, block_rows).unwrap());
                    assert_eq!(pairs.len(), rows * rows.saturating_sub(1) / 2);
                }
            }
        }
    }

    #[test]
    fn comparison_block_assignment_is_two_dimensional() {
        let first = assigned_blocks(300_000, 100_000, 0, 3)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(
            first,
            vec![
                (0..100_000, 0..100_000),
                (100_000..200_000, 100_000..200_000)
            ]
        );
        assert_eq!(pairwise_block_count(300_000, 100_000).unwrap(), 6);
        assert!(pairwise_block_count(usize::MAX, 1).is_err());
        assert!(pairwise_block_count(1, 0).is_err());
        assert!(assigned_blocks(1, 1, 0, 0).is_err());
        assert!(assigned_blocks(1, 1, 1, 1).is_err());
    }
}
