// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stable-partition mapping semantics, independent of dataset lineage traversal.

use super::row_map::RowMapReader;
use crate::scalar::IndexStore;
use async_trait::async_trait;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::fragment_reuse::MappingReader;
use lance_core::{Error, Result};
use lance_table::format::pb::fragment_reuse_index_details::FragmentDigest;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// Immutable file name within the mapping's independently owned directory.
pub const MAPPING_FILE: &str = "stable_partition.lance";

/// A mapping reader that opens labels only when addresses need translation.
pub struct StablePartitionMapping {
    store: Arc<dyn IndexStore>,
    reader: OnceCell<RowMapReader>,
    sources: HashMap<u32, (u64, u64)>,
    destinations: Vec<FragmentDigest>,
    total_rows: u64,
}

impl std::fmt::Debug for StablePartitionMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StablePartitionMapping")
            .field("sources", &self.sources)
            .field("destinations", &self.destinations)
            .finish_non_exhaustive()
    }
}

impl StablePartitionMapping {
    /// Bind the file store to source scan order and destination label order.
    /// The store resolves the dataset base; this reader does not interpret manifests.
    pub fn try_new(
        store: Arc<dyn IndexStore>,
        sources: Vec<FragmentDigest>,
        destinations: Vec<FragmentDigest>,
    ) -> Result<Self> {
        let mut total_rows = 0_u64;
        let mut offsets = HashMap::with_capacity(sources.len());
        for source in sources {
            if source.id >= u64::from(RowAddress::TOMBSTONE_FRAG)
                || source.num_deleted_rows > source.physical_rows
                || source.physical_rows > u32::MAX as u64
                || offsets
                    .insert(source.id as u32, (total_rows, source.physical_rows))
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
        let destination_fragments: RoaringBitmap =
            destinations.iter().map(|f| f.id as u32).collect();
        if destinations.len() > u16::MAX as usize
            || destination_fragments.len() != destinations.len() as u64
            || destinations.iter().any(|f| {
                f.id >= u64::from(RowAddress::TOMBSTONE_FRAG)
                    || f.physical_rows > u32::MAX as u64
                    || f.num_deleted_rows != 0
            })
        {
            return Err(Error::invalid_input(
                "invalid stable-partition destination layout",
            ));
        }
        Ok(Self {
            store,
            reader: OnceCell::new(),
            sources: offsets,
            destinations,
            total_rows,
        })
    }
}

impl DeepSizeOf for StablePartitionMapping {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.sources.capacity() * (std::mem::size_of::<(u32, (u64, u64))>() + 1)
            + self.destinations.capacity() * std::mem::size_of::<FragmentDigest>()
            + self
                .reader
                .get()
                .map_or(0, |reader| reader.counts().deep_size_of())
        // The IndexStore and file metadata belong to the session's metadata cache.
    }
}

#[async_trait]
impl MappingReader for StablePartitionMapping {
    async fn remap_row_id(&self, row_id: u64) -> Result<Option<u64>> {
        // Reuse the block reader for a single address too.
        let mapped = self.remap_row_ids(&[row_id]).await?;
        mapped
            .into_iter()
            .next()
            .ok_or_else(|| Error::internal("stable-partition mapping omitted the requested row"))
    }

    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        let mut output = vec![None; row_ids.len()];
        let mut requests = Vec::with_capacity(row_ids.len());
        for (position, &row_id) in row_ids.iter().enumerate() {
            let address = RowAddress::from(row_id);
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
                    Some(
                        RowAddress::new_from_parts(
                            self.destinations[label as usize].id as u32,
                            destination_offset,
                        )
                        .into(),
                    )
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
    async fn mapping_single_and_batch_translation_are_lazy() {
        let directory = TempDir::default();
        let (object_store, path) = ObjectStore::from_uri(directory.obj_path().as_ref())
            .await
            .unwrap();
        let store = Arc::new(LanceIndexStore::new(
            object_store,
            path,
            Arc::new(LanceCache::with_capacity(1024 * 1024)),
        ));
        let source = FragmentDigest {
            id: 1,
            physical_rows: 5,
            num_deleted_rows: 1,
        };
        let destinations = vec![
            FragmentDigest {
                id: 2,
                physical_rows: 2,
                num_deleted_rows: 0,
            },
            FragmentDigest {
                id: 3,
                physical_rows: 2,
                num_deleted_rows: 0,
            },
        ];
        let mapping =
            StablePartitionMapping::try_new(store.clone(), vec![source], destinations).unwrap();
        assert!(mapping.remap_row_ids(&[]).await.unwrap().is_empty());
        assert!(mapping.reader.get().is_none());
        // Empty input must not open the file.
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
        let addr = |fragment, offset| u64::from(RowAddress::new_from_parts(fragment, offset));
        assert_eq!(
            mapping
                .remap_row_ids(&[addr(1, 4), addr(1, 1), addr(1, 0), addr(1, 0), addr(1, 2)])
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
        assert_eq!(
            mapping.remap_row_id(addr(1, 4)).await.unwrap(),
            Some(addr(2, 1))
        );
        assert_eq!(mapping.remap_row_id(addr(1, 1)).await.unwrap(), None);
        for (row_id, message) in [
            (addr(9, 0), "outside stable-partition sources"),
            (addr(1, 5), "exceeds source length"),
        ] {
            for error in [
                mapping.remap_row_id(row_id).await.unwrap_err(),
                mapping.remap_row_ids(&[row_id]).await.unwrap_err(),
            ] {
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(error.to_string().contains(message));
            }
        }
    }
}
