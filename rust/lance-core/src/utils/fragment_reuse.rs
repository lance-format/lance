// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Readers for a single fragment rewrite. Graph traversal belongs to the FRI reader.

use std::sync::Arc;

use async_trait::async_trait;
use roaring::RoaringBitmap;

use super::address::RowAddress;
use super::row_addr_remap::RowAddrRemap;
use crate::{Error, Result};

/// The read contract for one mapping, independent of index types and manifests.
///
/// Coverage describes complete fragments, not the rows matched by a query.
/// Translation accepts source addresses in arbitrary order, including duplicates;
/// results preserve input positions and `None` denotes a deleted source row.
/// Implementations must reject addresses outside their source fragments.
///
/// ```
/// # use lance_core::utils::fragment_reuse::MappingReader;
/// # use lance_core::utils::address::RowAddress;
/// # async fn example(reader: &dyn MappingReader) -> lance_core::Result<()> {
/// let input = [RowAddress::new_from_parts(1, 0)];
/// let output = reader.translate(&input).await?;
/// assert_eq!(output.len(), input.len());
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait MappingReader: Send + Sync + std::fmt::Debug {
    /// Return destinations completely covered by the supplied source coverage.
    /// The input may include unrelated fragments. Partial coverage must never
    /// be advertised as complete. This operation reads metadata only.
    fn coverage(&self, covered_sources: &RoaringBitmap) -> RoaringBitmap;

    /// Translate one batch of source addresses; only mapping implementations
    /// that require external data perform asynchronous IO.
    async fn translate(&self, addresses: &[RowAddress]) -> Result<Vec<Option<RowAddress>>>;
}

/// Ordered-compaction adapter for the existing bitmap/rank remap algorithm.
/// This adapter does not replace the legacy FRI reader or writer.
#[derive(Debug)]
pub struct OrderedCompactionMapping {
    remap: Arc<RowAddrRemap>,
    sources: RoaringBitmap,
    destinations: RoaringBitmap,
}

impl OrderedCompactionMapping {
    /// Bind a validated compaction remap to its source and destination fragments.
    /// Callers validate the rewrite layout when constructing `remap`.
    pub fn new(
        remap: Arc<RowAddrRemap>,
        sources: RoaringBitmap,
        destinations: RoaringBitmap,
    ) -> Self {
        Self {
            remap,
            sources,
            destinations,
        }
    }
}

#[async_trait]
impl MappingReader for OrderedCompactionMapping {
    fn coverage(&self, covered_sources: &RoaringBitmap) -> RoaringBitmap {
        if self.sources.is_subset(covered_sources) {
            self.destinations.clone()
        } else {
            RoaringBitmap::new()
        }
    }

    async fn translate(&self, addresses: &[RowAddress]) -> Result<Vec<Option<RowAddress>>> {
        addresses
            .iter()
            .map(|&address| {
                if !self.sources.contains(address.fragment_id()) {
                    return Err(Error::invalid_input(format!(
                        "address {address} is outside compaction sources"
                    )));
                }
                self.remap
                    .get(address.into())
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "address {address} is outside compaction mapping"
                        ))
                    })
                    .map(|mapped| mapped.map(RowAddress::from))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::row_addr_remap::GroupInputWithLayout;
    use roaring::RoaringTreemap;

    #[tokio::test]
    async fn compaction_reader_preserves_legacy_mapping_semantics() {
        let address = |fragment, offset| RowAddress::new_from_parts(fragment, offset);
        let remap = Arc::new(
            RowAddrRemap::compact_with_layout([GroupInputWithLayout {
                rewritten_old_row_addrs: RoaringTreemap::from_iter([
                    address(1, 0).into(),
                    address(1, 2).into(),
                    address(2, 0).into(),
                ]),
                old_frags: vec![(1, 3), (2, 1)],
                new_frags: vec![(3, 2), (4, 1)],
            }])
            .unwrap(),
        );
        let reader = OrderedCompactionMapping::new(
            remap.clone(),
            [1, 2].into_iter().collect(),
            [3, 4].into_iter().collect(),
        );
        assert!(reader.coverage(&[1].into_iter().collect()).is_empty());
        assert_eq!(
            reader.coverage(&[1, 2, 99].into_iter().collect()),
            [3, 4].into_iter().collect()
        );
        let input = [address(2, 0), address(1, 1), address(1, 0), address(1, 0)];
        let expected: Vec<_> = input
            .iter()
            .map(|&a| remap.get(a.into()).unwrap().map(RowAddress::from))
            .collect();
        assert_eq!(reader.translate(&input).await.unwrap(), expected);
        assert!(reader.translate(&[]).await.unwrap().is_empty());
        let error = reader.translate(&[address(99, 0)]).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
    }
}
