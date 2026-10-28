// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Request-sized integer matching for uncovered scan batches.

use arrow_array::{Array, UInt64Array};
use lance_core::Result;
use lance_core::utils::address::RowAddress;
use rustc_hash::FxHashMap;

use super::assign_unique_address;

pub(super) struct UInt64KeyPositions {
    first: FxHashMap<u64, usize>,
    // Link duplicate inputs in one buffer instead of allocating a Vec per key.
    next: Vec<Option<usize>>,
}

impl UInt64KeyPositions {
    pub(super) fn new(keys: &UInt64Array) -> Self {
        let mut first =
            FxHashMap::with_capacity_and_hasher(keys.len() - keys.null_count(), Default::default());
        let mut next = vec![None; keys.len()];
        for position in (0..keys.len()).rev() {
            if keys.is_valid(position) {
                next[position] = first.insert(keys.value(position), position);
            }
        }
        Self { first, next }
    }

    pub(super) fn assign_matches(
        &self,
        values: &UInt64Array,
        row_addresses: &UInt64Array,
        addresses: &mut [Option<RowAddress>],
        column: &str,
    ) -> Result<()> {
        let mut assign = |first: usize, offset: usize| -> Result<()> {
            if row_addresses.is_null(offset) {
                return Ok(());
            }
            let address = RowAddress::from(row_addresses.value(offset));
            let mut position = Some(first);
            while let Some(key) = position {
                assign_unique_address(addresses, key, address, column)?;
                position = self.next[key];
            }
            Ok(())
        };

        // Only actual batch order is a sufficient proof: compaction can spill
        // arbitrarily permuted IDs, and a user UInt64 column need not be sorted.
        if values.null_count() == 0
            && self.first.len() < values.len()
            && values.values().is_sorted()
        {
            let min = values.value(0);
            let max = values.value(values.len() - 1);
            let candidates = self
                .first
                .keys()
                .filter(|&&value| value >= min && value <= max)
                .count();
            // Prefer a scan for dense probes or small batches. Counting only
            // in-range keys also avoids binary searches for absent batch ranges.
            if candidates.saturating_mul(values.len().ilog2() as usize + 1) < values.len() {
                for (&value, &first) in &self.first {
                    if value < min || value > max {
                        continue;
                    }
                    let mut offset = values.values().partition_point(|&item| item < value);
                    while offset < values.len() && values.value(offset) == value {
                        assign(first, offset)?;
                        offset += 1;
                    }
                }
                return Ok(());
            }
        }

        if self.first.len() == 1
            && let Some((&key, &first)) = self.first.iter().next()
        {
            for (offset, value) in values.iter().enumerate() {
                if value == Some(key) {
                    assign(first, offset)?;
                }
            }
        } else {
            for (offset, value) in values.iter().enumerate() {
                if let Some(value) = value
                    && let Some(&first) = self.first.get(&value)
                {
                    assign(first, offset)?;
                }
            }
        }
        Ok(())
    }
}
