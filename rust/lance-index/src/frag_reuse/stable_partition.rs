// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stable-partition mapping semantics, independent of dataset lineage traversal.

use super::row_map::RowMapReader;
use crate::scalar::IndexStore;
use async_trait::async_trait;
use lance_core::utils::address::RowAddress;
use lance_core::utils::fragment_reuse::MappingReader;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// Immutable file name within the mapping's independently owned directory.
pub const MAPPING_FILE: &str = "stable_partition.lance";

/// Physical fragment layout, including deleted rows in source fragments.
#[derive(Debug, Clone, Copy)]
pub struct FragmentLayout {
    /// Dataset fragment identifier.
    pub id: u32,
    /// Physical row count recorded when the mapping was created.
    pub physical_rows: u64,
}

/// A mapping reader that opens labels only when addresses need translation.
/// Coverage uses fragment metadata and never opens the external file.
pub struct StablePartitionMapping {
    store: Arc<dyn IndexStore>,
    reader: OnceCell<RowMapReader>,
    sources: HashMap<u32, (u64, u64)>,
    source_fragments: RoaringBitmap,
    destinations: Vec<FragmentLayout>,
    destination_fragments: RoaringBitmap,
    total_rows: u64,
}

impl std::fmt::Debug for StablePartitionMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StablePartitionMapping")
            .field("sources", &self.source_fragments)
            .field("destinations", &self.destinations)
            .finish_non_exhaustive()
    }
}

impl StablePartitionMapping {
    /// Bind the file store to source scan order and destination label order.
    /// The store resolves the dataset base; this reader does not interpret manifests.
    pub fn try_new(
        store: Arc<dyn IndexStore>,
        sources: Vec<FragmentLayout>,
        destinations: Vec<FragmentLayout>,
    ) -> Result<Self> {
        let mut total_rows = 0_u64;
        let mut offsets = HashMap::with_capacity(sources.len());
        for source in sources {
            if source.physical_rows > u32::MAX as u64
                || offsets
                    .insert(source.id, (total_rows, source.physical_rows))
                    .is_some()
            {
                return Err(Error::invalid_input(format!(
                    "invalid or duplicate source fragment {} with {} rows",
                    source.id, source.physical_rows
                )));
            }
            total_rows = total_rows
                .checked_add(source.physical_rows)
                .ok_or_else(|| Error::invalid_input("source physical row count overflow"))?;
        }
        let destination_fragments: RoaringBitmap = destinations.iter().map(|f| f.id).collect();
        if destinations.len() > u16::MAX as usize
            || destination_fragments.len() != destinations.len() as u64
            || destinations
                .iter()
                .any(|f| f.physical_rows > u32::MAX as u64)
        {
            return Err(Error::invalid_input(
                "invalid stable-partition destination layout",
            ));
        }
        Ok(Self {
            store,
            reader: OnceCell::new(),
            source_fragments: offsets.keys().copied().collect(),
            sources: offsets,
            destinations,
            destination_fragments,
            total_rows,
        })
    }
}

#[async_trait]
impl MappingReader for StablePartitionMapping {
    fn coverage(&self, covered_sources: &RoaringBitmap) -> RoaringBitmap {
        if self.source_fragments.is_subset(covered_sources) {
            self.destination_fragments.clone()
        } else {
            RoaringBitmap::new()
        }
    }

    async fn translate(&self, addresses: &[RowAddress]) -> Result<Vec<Option<RowAddress>>> {
        let mut output = vec![None; addresses.len()];
        let mut requests = Vec::with_capacity(addresses.len());
        for (position, address) in addresses.iter().enumerate() {
            let &(base, rows) = self.sources.get(&address.fragment_id()).ok_or_else(|| {
                Error::invalid_input(format!(
                    "address {address} is outside stable-partition sources"
                ))
            })?;
            if u64::from(address.row_offset()) >= rows {
                return Err(Error::invalid_input(format!(
                    "address {address} exceeds source length {rows}"
                )));
            }
            requests.push((base + u64::from(address.row_offset()), position));
        }
        if requests.is_empty() {
            return Ok(output);
        }
        let reader = self
            .reader
            .get_or_try_init(|| async {
                let reader =
                    RowMapReader::open(self.store.open_index_file(MAPPING_FILE).await?).await?;
                let counts = reader.counts();
                if counts.total_rows() != self.total_rows
                    || counts.num_destinations() as usize != self.destinations.len()
                {
                    return Err(corrupt("row-map dimensions differ from transition digests"));
                }
                for (label, destination) in self.destinations.iter().enumerate() {
                    if u64::from(counts.total(label as u16)) != destination.physical_rows {
                        return Err(corrupt(format!(
                            "row-map total differs for destination {}",
                            destination.id
                        )));
                    }
                }
                Ok(reader)
            })
            .await?;
        requests.sort_unstable_by_key(|&(row, _)| row);
        let mut remaining = requests.as_slice();
        while let Some(&(first, _)) = remaining.first() {
            let counts = reader.counts();
            let block = counts.block_of(first);
            let range = counts.block_range(block);
            let end = remaining.partition_point(|&(row, _)| row < range.end);
            let (batch, rest) = remaining.split_at(end);
            let labels = reader.block_labels(block).await?;
            if labels.len() as u64 != range.end - range.start {
                return Err(corrupt(format!(
                    "row-map block {block} has an unexpected label count"
                )));
            }
            // One sweep per touched block: bounded label memory and no
            // repeated prefix scans for dense index pages or duplicates.
            let mut counters = counts.counters_at_block(block);
            let mut requested = batch.iter().peekable();
            for (offset, label) in labels.iter().enumerate() {
                let translated = if let Some(label) = label {
                    let counter = counters
                        .get_mut(label as usize)
                        .ok_or_else(|| corrupt(format!("invalid row-map label {label}")))?;
                    let destination_offset = *counter;
                    *counter = counter
                        .checked_add(1)
                        .ok_or_else(|| corrupt("row-map count overflow"))?;
                    Some(RowAddress::new_from_parts(
                        self.destinations[label as usize].id,
                        destination_offset,
                    ))
                } else {
                    None
                };
                let row = range.start + offset as u64;
                while let Some(&&(requested_row, position)) = requested.peek() {
                    if requested_row != row {
                        break;
                    }
                    output[position] = translated;
                    requested.next();
                }
            }
            if counters != counts.counters_at_block(block + 1) {
                return Err(corrupt(format!(
                    "row-map labels disagree with counts in block {block}"
                )));
            }
            remaining = rest;
        }

        Ok(output)
    }
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::corrupt_file_named(MAPPING_FILE, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use crate::scalar::lance_format::LanceIndexStore;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempDir;
    use lance_io::object_store::ObjectStore;

    #[tokio::test]
    async fn mapping_coverage_and_lazy_translation() {
        let directory = TempDir::default();
        let (object_store, path) = ObjectStore::from_uri(directory.obj_path().as_ref())
            .await
            .unwrap();
        let store = Arc::new(LanceIndexStore::new(
            object_store,
            path,
            Arc::new(LanceCache::with_capacity(1024 * 1024)),
        ));
        let source = FragmentLayout {
            id: 1,
            physical_rows: 5,
        };
        let destinations = vec![
            FragmentLayout {
                id: 2,
                physical_rows: 2,
            },
            FragmentLayout {
                id: 3,
                physical_rows: 2,
            },
        ];
        let mapping =
            StablePartitionMapping::try_new(store.clone(), vec![source], destinations).unwrap();
        assert_eq!(
            mapping.coverage(&[1].into_iter().collect()),
            [2, 3].into_iter().collect()
        );
        assert!(mapping.coverage(&RoaringBitmap::new()).is_empty());
        assert!(mapping.translate(&[]).await.unwrap().is_empty());
        assert!(mapping.reader.get().is_none());
        // No file exists until after the metadata-only operations above.
        let writer = store
            .new_index_file(MAPPING_FILE, RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer = RowMapWriter::try_new_with_block_rows(
            writer,
            vec![SourceRows {
                physical_rows: 5,
                deleted: Some([1].into_iter().collect()),
            }],
            2,
            2,
        )
        .unwrap();
        writer.append_labels(&[1, 0, 1, 0]).await.unwrap();
        writer.finish().await.unwrap();
        let addr = RowAddress::new_from_parts;
        assert_eq!(
            mapping
                .translate(&[addr(1, 4), addr(1, 1), addr(1, 0), addr(1, 0), addr(1, 2)])
                .await
                .unwrap(),
            vec![
                Some(addr(2, 1)),
                None,
                Some(addr(3, 0)),
                Some(addr(3, 0)),
                Some(addr(2, 0))
            ]
        );
        assert!(mapping.reader.get().is_some());
        let error = mapping.translate(&[addr(9, 0)]).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("outside stable-partition sources")
        );
    }
}
