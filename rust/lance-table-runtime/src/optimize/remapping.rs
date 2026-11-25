// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index remapping contracts and physical row-address translation.

use std::collections::HashMap;

use async_trait::async_trait;
use lance_core::Result;
use lance_core::utils::address::RowAddress;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_table::format::{Fragment, IndexFile};
use lance_table::system_index::frag_reuse::FragDigest;
use lance_table::transaction::RewriteGroup;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The result of remapping an index
#[derive(Debug, Clone, PartialEq)]
pub enum RemapResult {
    // Index could not be remapped, drop it
    Drop,
    // No remapping is needed, keep the index as-is
    Keep(Uuid),
    // Index was remapped, return the new index
    Remapped(RemappedIndex),
}

/// A remapped index
#[derive(Debug, Clone, PartialEq)]
pub struct RemappedIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
    pub index_details: prost_types::Any,
    pub index_version: u32,
    /// List of files in the index with their sizes.
    pub files: Option<Vec<IndexFile>>,
}

/// When compaction runs the row ids will change.  This typically means that
/// indices will need to be remapped.  The details of how this happens are not
/// a part of the compaction process and so a trait is defined here to allow
/// for inversion of control.
#[async_trait]
pub trait IndexRemapper: Send + Sync {
    async fn remap_indices(
        &self,
        index_map: RowAddrRemap,
        affected_fragment_ids: &[u64],
    ) -> Result<Vec<RemappedIndex>>;

    /// Remap with the completed rewrite groups that determine which source
    /// segments must be published as one atomic replacement. Existing remappers
    /// retain their independent-segment behavior through the default method.
    async fn remap_indices_with_context(
        &self,
        index_map: RowAddrRemap,
        affected_fragment_ids: &[u64],
        _rewrite_groups: &[RewriteGroup],
    ) -> Result<Vec<RemappedIndex>> {
        self.remap_indices(index_map, affected_fragment_ids).await
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct IgnoreRemap {}

#[async_trait]
impl IndexRemapper for IgnoreRemap {
    async fn remap_indices(&self, _: RowAddrRemap, _: &[u64]) -> Result<Vec<RemappedIndex>> {
        Ok(Vec::new())
    }
}

/// Iterator that yields row_addrs that are in the given fragments but not in
/// the given row_addrs iterator.
struct MissingAddrs<'a, I: Iterator<Item = u64>> {
    row_addrs: I,
    expected_row_addr: u64,
    current_fragment_idx: usize,
    last: Option<u64>,
    fragments: &'a Vec<FragDigest>,
}

impl<'a, I: Iterator<Item = u64>> MissingAddrs<'a, I> {
    /// row_addrs must be sorted in the same order in which the rows would be
    /// found by scanning fragments in the order they are presented in.
    /// fragments is not guaranteed to be sorted by id.
    fn new(row_addrs: I, fragments: &'a Vec<FragDigest>) -> Self {
        assert!(!fragments.is_empty());
        let first_frag = &fragments[0];
        Self {
            row_addrs,
            expected_row_addr: first_frag.id * RowAddress::FRAGMENT_SIZE,
            current_fragment_idx: 0,
            last: None,
            fragments,
        }
    }
}

impl<I: Iterator<Item = u64>> Iterator for MissingAddrs<'_, I> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_fragment_idx >= self.fragments.len() {
                return None;
            }
            let val = if let Some(last) = self.last {
                self.last = None;
                last
            } else {
                // The tombstone fragment id cannot match a real fragment, so all
                // remaining expected addresses are reported as missing.
                self.row_addrs.next().unwrap_or(RowAddress::TOMBSTONE_ROW)
            };

            let current_fragment = &self.fragments[self.current_fragment_idx];
            let frag = val / RowAddress::FRAGMENT_SIZE;
            let expected_row_addr = self.expected_row_addr;
            self.expected_row_addr += 1;

            let current_physical_rows = current_fragment.physical_rows;
            if (self.expected_row_addr % RowAddress::FRAGMENT_SIZE) == current_physical_rows as u64
            {
                self.current_fragment_idx += 1;
                if self.current_fragment_idx < self.fragments.len() {
                    self.expected_row_addr =
                        self.fragments[self.current_fragment_idx].id * RowAddress::FRAGMENT_SIZE;
                }
            }
            if frag != current_fragment.id {
                self.last = Some(val);
                return Some(expected_row_addr);
            }
            if val != expected_row_addr {
                self.last = Some(val);
                return Some(expected_row_addr);
            }
        }
    }
}

pub fn transpose_row_addrs(
    row_addrs: RoaringTreemap,
    old_fragments: &[Fragment],
    new_fragments: &[Fragment],
) -> HashMap<u64, Option<u64>> {
    let old_frag_digests: Vec<FragDigest> = old_fragments.iter().map(|frag| frag.into()).collect();
    let new_frag_digests: Vec<FragDigest> = new_fragments.iter().map(|frag| frag.into()).collect();
    transpose_row_ids_from_digest(row_addrs, &old_frag_digests, &new_frag_digests)
}

pub fn transpose_row_ids_from_digest(
    row_addrs: RoaringTreemap,
    old_fragments: &Vec<FragDigest>,
    new_fragments: &[FragDigest],
) -> HashMap<u64, Option<u64>> {
    let new_addrs = new_fragments.iter().flat_map(|frag| {
        (0..frag.physical_rows as u32).map(|offset| {
            Some(u64::from(RowAddress::new_from_parts(
                frag.id as u32,
                offset,
            )))
        })
    });
    // The hashmap will have an entry for each row addr to map plus all rows that
    // were deleted.
    let expected_size = row_addrs.len() as usize
        + old_fragments
            .iter()
            .map(|frag| frag.num_deleted_rows)
            .sum::<usize>();
    // We expect row addrs to be unique, so we should already not get many collisions.
    // The default hasher is designed to be resistance to DoS attacks, which is
    // more than we need for this use case.
    let mut mapping: HashMap<u64, Option<u64>> = HashMap::with_capacity(expected_size);
    mapping.extend(row_addrs.iter().zip(new_addrs));
    MissingAddrs::new(row_addrs.into_iter(), old_fragments).for_each(|addr| {
        mapping.insert(addr, None);
    });
    mapping
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_indices() {
        // Sanity test to make sure MissingIds works.  Does not test actual functionality so
        // feel free to remove if it becomes inconvenient
        let frags = vec![
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];
        let rows = [(0, 1), (0, 3), (0, 4), (3, 0), (3, 2)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());

        let missing = MissingAddrs::new(rows, &frags).collect::<Vec<_>>();
        let expected_missing = [(0, 0), (0, 2), (3, 1)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(missing, expected_missing);
    }

    #[test]
    fn test_missing_ids() {
        // test with missing first row
        // test with missing last row
        // test fragment ids out of order

        let fragments = vec![
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 1,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];

        // Written as pairs of (fragment_id, offset)
        let row_addrs = vec![
            (0, 1),
            (0, 3),
            (0, 4),
            (3, 0),
            (3, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ];
        let row_addrs = row_addrs
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());
        let result = MissingAddrs::new(row_addrs, &fragments).collect::<Vec<_>>();

        let expected = vec![(0, 0), (0, 2), (3, 1)];
        let expected = expected
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(result, expected);
    }
}
