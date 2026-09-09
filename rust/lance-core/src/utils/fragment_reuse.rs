// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Readers for a single fragment rewrite. Graph traversal belongs to the FRI reader.

use std::sync::Arc;

use async_trait::async_trait;

use super::row_addr_remap::RowAddrRemap;
use crate::deepsize::{Context, DeepSizeOf};
use crate::{Error, Result};

/// The read contract for one mapping, independent of index types and manifests.
///
/// Row IDs are physical addresses. Results preserve input positions and `None`
/// denotes a deleted source row. Addresses outside the source fragments, including
/// out-of-range offsets, return [`Error::InvalidInput`]. Invalid persisted mapping
/// contents return [`Error::CorruptFile`]; storage errors propagate to the caller.
#[async_trait]
pub trait MappingReader: Send + Sync + std::fmt::Debug + DeepSizeOf {
    /// Remap one physical row ID, or return `None` if the row was deleted.
    async fn remap_row_id(&self, row_id: u64) -> Result<Option<u64>> {
        let mapped = self.remap_row_ids(&[row_id]).await?;
        match mapped.as_slice() {
            [result] => Ok(*result),
            _ => Err(Error::internal(format!(
                "mapping returned {} results for one row ID",
                mapped.len(),
            ))),
        }
    }

    /// Remap a batch, preserving input order and duplicates, with one result per input.
    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>>;
}

/// Ordered-compaction adapter for one rewrite group's existing bitmap/rank remap.
/// This adapter does not replace the legacy FRI reader or writer.
#[derive(Debug)]
pub struct OrderedCompactionMapping {
    remap: Arc<RowAddrRemap>,
}

impl OrderedCompactionMapping {
    /// Use a remap whose fragment layout was validated when it was constructed.
    pub fn new(remap: Arc<RowAddrRemap>) -> Self {
        Self { remap }
    }

    fn map_row_id(&self, row_id: u64) -> Result<Option<u64>> {
        self.remap.get(row_id).ok_or_else(|| {
            Error::invalid_input(format!("row ID {row_id} is outside compaction mapping"))
        })
    }
}

impl DeepSizeOf for OrderedCompactionMapping {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.remap.deep_size_of_children(context)
    }
}

#[async_trait]
impl MappingReader for OrderedCompactionMapping {
    async fn remap_row_id(&self, row_id: u64) -> Result<Option<u64>> {
        self.map_row_id(row_id)
    }

    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        row_ids
            .iter()
            .map(|&row_id| self.map_row_id(row_id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::address::RowAddress;
    use crate::utils::row_addr_remap::GroupInputWithLayout;
    use roaring::RoaringTreemap;

    #[tokio::test]
    async fn compaction_reader_preserves_legacy_mapping_semantics() {
        let address = |fragment, offset| u64::from(RowAddress::new_from_parts(fragment, offset));
        let remap = Arc::new(
            RowAddrRemap::compact_with_layout([GroupInputWithLayout {
                rewritten_old_row_addrs: RoaringTreemap::from_iter([
                    address(1, 0),
                    address(1, 2),
                    address(2, 0),
                ]),
                old_frags: vec![(1, 3), (2, 1)],
                new_frags: vec![(3, 2), (4, 1)],
            }])
            .unwrap(),
        );
        let reader = OrderedCompactionMapping::new(remap.clone());
        let input = [address(2, 0), address(1, 1), address(1, 0), address(1, 0)];
        let expected: Vec<_> = input.iter().map(|&a| remap.get(a).unwrap()).collect();
        assert_eq!(reader.remap_row_ids(&input).await.unwrap(), expected);
        for (&row_id, &mapped) in input.iter().zip(&expected) {
            assert_eq!(reader.remap_row_id(row_id).await.unwrap(), mapped);
        }
        assert!(reader.remap_row_ids(&[]).await.unwrap().is_empty());
        for invalid in [address(99, 0), address(1, 3)] {
            for error in [
                reader.remap_row_id(invalid).await.unwrap_err(),
                reader
                    .remap_row_ids(&[input[0], invalid])
                    .await
                    .unwrap_err(),
            ] {
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(error.to_string().contains("outside compaction mapping"));
            }
        }
    }
}
