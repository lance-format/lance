// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Encoded-window admission; cache, postings and decode scratch are separate.

use super::{MAX_PARTITION_WINDOW_BYTES, MAX_QUERY_WINDOW_BYTES};
use lance_core::{Error, Result};

// Bound task metadata independently of the size of an encoded window.
const MAX_ACTIVE_PARTITIONS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub(super) struct WindowDemand {
    /// Largest complete window, which must fit even without prediction.
    pub(super) current: u64,
    /// Complete encoded value; a small stream cannot use a larger allowance.
    pub(super) total: u64,
}

pub(super) struct SearchResources {
    pub(super) active_partitions: usize,
    pub(super) partition_bytes: u64,
}

/// Target current plus next windows only where the stream has that much data.
pub(super) fn working_bytes(demands: &[WindowDemand]) -> Result<u64> {
    demands.iter().try_fold(0_u64, |sum, demand| {
        let remaining = demand.total.checked_sub(demand.current).ok_or_else(|| {
            Error::index("selective position window exceeds its complete encoded value")
        })?;
        let current_and_next = demand.current + demand.current.min(remaining);
        sum.checked_add(current_and_next)
            .ok_or_else(|| Error::index("selective position working set overflows u64"))
    })
}

pub(super) fn search_resources(
    cpus: usize,
    partitions: usize,
    working_bytes: u64,
) -> Result<SearchResources> {
    if cpus == 0 || partitions == 0 || working_bytes == 0 {
        return Err(Error::invalid_input(format!(
            "selective resource inputs must be positive: cpus={cpus}, partitions={partitions}, working_bytes={working_bytes}"
        )));
    }
    // Prediction is optional. When two windows per stream exceed the cap,
    // retain the cap and admit current windows separately before loading data.
    let target = working_bytes.min(MAX_PARTITION_WINDOW_BYTES);
    let by_bytes = (MAX_QUERY_WINDOW_BYTES / target) as usize;
    let active_partitions = cpus
        .min(partitions)
        .min(MAX_ACTIVE_PARTITIONS)
        .min(by_bytes);
    Ok(SearchResources {
        active_partitions,
        partition_bytes: (MAX_QUERY_WINDOW_BYTES / active_partitions as u64)
            .min(MAX_PARTITION_WINDOW_BYTES),
    })
}

/// Reserve a complete current window for each distinct stream, then fairly
/// share the remainder. Return unused small-stream quotas to larger streams.
/// None selects ordinary search before postings or a score floor are consumed.
pub(super) fn allocate_bytes(demands: &[WindowDemand], budget: u64) -> Result<Option<Vec<u64>>> {
    let mut allocated = Vec::with_capacity(demands.len());
    let mut remaining = budget;
    let mut spare = Vec::with_capacity(demands.len());
    for (index, demand) in demands.iter().enumerate() {
        let capacity = demand.total.checked_sub(demand.current).ok_or_else(|| {
            Error::index("selective position window exceeds its complete encoded value")
        })?;
        let Some(bytes) = remaining.checked_sub(demand.current) else {
            return Ok(None);
        };
        remaining = bytes;
        allocated.push(demand.current);
        spare.push((index, capacity));
    }
    spare.sort_unstable_by_key(|&(index, capacity)| (capacity, index));
    for (ordinal, &(index, capacity)) in spare.iter().enumerate() {
        let extra = capacity.min(remaining / (spare.len() - ordinal) as u64);
        allocated[index] += extra;
        remaining -= extra;
    }
    Ok(Some(allocated))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demand(current: u64, total: u64) -> WindowDemand {
        WindowDemand { current, total }
    }

    #[rstest::rstest]
    fn selective_phrase_resources_obey_cpu_and_byte_caps(
        #[values(1, 2, 8, 16, 64)] cpus: usize,
        #[values(1, 3, 19, 1000)] partitions: usize,
        #[values(1, 8_388_608, 20_971_520, 67_108_864)] working: u64,
    ) {
        let resources = search_resources(cpus, partitions, working).unwrap();
        assert!(
            (1..=cpus.min(partitions).min(MAX_ACTIVE_PARTITIONS))
                .contains(&resources.active_partitions)
        );
        assert!(resources.partition_bytes <= MAX_PARTITION_WINDOW_BYTES);
        assert!(
            resources.partition_bytes * resources.active_partitions as u64
                <= MAX_QUERY_WINDOW_BYTES
        );
        assert!(resources.partition_bytes >= working.min(MAX_PARTITION_WINDOW_BYTES));
    }

    #[test]
    fn selective_phrase_resources_use_stream_sizes_and_keep_cpu1() {
        let mib = 1024 * 1024;
        let small = demand(mib / 4, mib / 4);
        let large = demand(2 * mib, 100 * mib);
        let sparse = working_bytes(&[small, small, small, small, large]).unwrap();
        let dense = working_bytes(&[large; 5]).unwrap();
        assert_eq!(sparse, 5 * mib);
        assert_eq!(dense, 20 * mib);
        assert!(
            search_resources(16, 19, sparse).unwrap().active_partitions
                > search_resources(16, 19, dense).unwrap().active_partitions
        );
        let single = search_resources(1, 19, sparse).unwrap();
        assert_eq!(single.active_partitions, 1);
        assert_eq!(single.partition_bytes, MAX_PARTITION_WINDOW_BYTES);
        for (cpus, parts, bytes) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert!(search_resources(cpus, parts, bytes).is_err());
        }
    }

    #[rstest::rstest]
    #[case::small_and_large(vec![demand(1, 1), demand(2, 100)], 16, Some(vec![1, 15]))]
    #[case::equal(vec![demand(2, 100); 2], 8, Some(vec![4, 4]))]
    #[case::large_current(vec![demand(7, 100), demand(1, 100)], 10, Some(vec![8, 2]))]
    #[case::single_distinct_stream(vec![demand(2, 100)], 16, Some(vec![16]))]
    #[case::all_complete(vec![demand(1, 1), demand(2, 3)], 100, Some(vec![1, 3]))]
    #[case::empty_value(vec![demand(0, 0), demand(2, 100)], 16, Some(vec![0, 16]))]
    #[case::no_streams(vec![], 16, Some(vec![]))]
    #[case::insufficient(vec![demand(7, 100), demand(4, 100)], 10, None)]
    #[case::largest_value(vec![demand(u64::MAX, u64::MAX)], u64::MAX, Some(vec![u64::MAX]))]
    fn selective_phrase_window_allocation(
        #[case] demands: Vec<WindowDemand>,
        #[case] budget: u64,
        #[case] expected: Option<Vec<u64>>,
    ) {
        assert_eq!(allocate_bytes(&demands, budget).unwrap(), expected);
    }

    #[rstest::rstest]
    fn selective_phrase_window_allocations_fit_whole_windows(
        #[values(8, 31, 128, 1024)] budget: u64,
        #[values(1, 4, 16)] count: usize,
    ) {
        let demands = (0..count)
            .map(|i| demand(1 + i as u64, (1 + i as u64) * 5))
            .collect::<Vec<_>>();
        let minima = demands.iter().map(|d| d.current).sum::<u64>();
        let allocated = allocate_bytes(&demands, budget).unwrap();
        assert_eq!(allocated.is_some(), minima <= budget);
        if let Some(allocated) = allocated {
            assert!(allocated.iter().sum::<u64>() <= budget);
            assert!(
                demands
                    .iter()
                    .zip(&allocated)
                    .all(|(d, &bytes)| (d.current..=d.total).contains(&bytes))
            );
            // Any unallocated byte is explained by complete streams, not a
            // quota stranded at a small term while another can still use it.
            assert!(
                allocated.iter().sum::<u64>() == budget
                    || demands
                        .iter()
                        .zip(&allocated)
                        .all(|(d, &bytes)| bytes == d.total)
            );
        }
    }

    #[test]
    fn selective_phrase_window_demands_reject_corrupt_sizes_and_overflow() {
        assert!(working_bytes(&[demand(2, 1)]).is_err());
        assert!(allocate_bytes(&[demand(2, 1)], 100).is_err());
        assert_eq!(
            working_bytes(&[demand(u64::MAX, u64::MAX)]).unwrap(),
            u64::MAX
        );
        assert!(working_bytes(&[demand(u64::MAX, u64::MAX), demand(1, 1)]).is_err());
    }
}
