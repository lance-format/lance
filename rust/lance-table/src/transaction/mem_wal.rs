// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL Index helpers used by transaction processing.
//!
//! Moved from `lance::index::mem_wal` so action / commit code can reach them
//! without depending on the top-level `lance` crate. The original module in
//! `lance` re-exports these for source compatibility.
//!
//! The MemWAL Index stores:
//! - Configuration (shard_specs, maintained_indexes)
//! - Merge progress (merged_generations per shard)
//! - Shard state snapshots (eventually consistent)
//!
//! Writers no longer update the index on every write. Instead, they update
//! shard manifests directly. This module provides functions to:
//! - Load the MemWAL index
//! - Update merged generations (called during merge-insert commits)

use std::sync::Arc;

use crate::format::{IndexMetadata, pb};
use crate::system_index::mem_wal::{
    MEM_WAL_INDEX_NAME, MemWalIndex, MemWalIndexDetails, MergedGeneration,
};
use lance_core::{Error, Result};
use uuid::Uuid;

/// Load MemWalIndexDetails from an IndexMetadata.
pub fn load_mem_wal_index_details(index: IndexMetadata) -> Result<MemWalIndexDetails> {
    if let Some(details_any) = index.index_details.as_ref() {
        if !details_any.type_url.ends_with("MemWalIndexDetails") {
            return Err(Error::index(format!(
                "Index details is not for the MemWAL index, but {}",
                details_any.type_url
            )));
        }

        Ok(MemWalIndexDetails::try_from(
            details_any.to_msg::<pb::MemWalIndexDetails>()?,
        )?)
    } else {
        Err(Error::index("Index details not found for the MemWAL index"))
    }
}

/// Open the MemWAL index from its metadata.
pub fn open_mem_wal_index(index: IndexMetadata) -> Result<Arc<MemWalIndex>> {
    Ok(Arc::new(MemWalIndex::new(load_mem_wal_index_details(
        index,
    )?)))
}

/// Update merged_generations in the MemWAL index.
/// This is called during merge-insert commits to atomically record which
/// generations have been merged to the base table.
pub fn update_mem_wal_index_merged_generations(
    indices: &mut Vec<IndexMetadata>,
    dataset_version: u64,
    new_merged_generations: Vec<MergedGeneration>,
) -> Result<()> {
    if new_merged_generations.is_empty() {
        return Ok(());
    }

    let pos = indices
        .iter()
        .position(|idx| idx.name == MEM_WAL_INDEX_NAME);

    let new_meta = if let Some(pos) = pos {
        let current_meta = indices.remove(pos);
        let mut details = load_mem_wal_index_details(current_meta)?;

        // Update merged_generations - for each shard, keep the higher generation
        for new_mg in new_merged_generations {
            if let Some(existing) = details
                .merged_generations
                .iter_mut()
                .find(|mg| mg.shard_id == new_mg.shard_id)
            {
                if new_mg.generation > existing.generation {
                    existing.generation = new_mg.generation;
                }
            } else {
                details.merged_generations.push(new_mg);
            }
        }

        new_mem_wal_index_meta(dataset_version, details)?
    } else {
        // Create new MemWAL index with just the merged generations
        let details = MemWalIndexDetails {
            merged_generations: new_merged_generations,
            ..Default::default()
        };
        new_mem_wal_index_meta(dataset_version, details)?
    };

    indices.push(new_meta);
    Ok(())
}

/// Create a new MemWAL index metadata entry.
pub fn new_mem_wal_index_meta(
    dataset_version: u64,
    details: MemWalIndexDetails,
) -> Result<IndexMetadata> {
    Ok(IndexMetadata {
        uuid: Uuid::new_v4(),
        name: MEM_WAL_INDEX_NAME.to_string(),
        fields: vec![],
        dataset_version,
        fragment_bitmap: None,
        index_details: Some(Arc::new(prost_types::Any::from_msg(
            &pb::MemWalIndexDetails::from(&details),
        )?)),
        index_version: 0,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        // Memory WAL index is inline (no files)
        files: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_merged_generations() {
        let mut indices = Vec::new();
        let shard1 = Uuid::new_v4();
        let shard2 = Uuid::new_v4();

        // First update - creates new index
        update_mem_wal_index_merged_generations(
            &mut indices,
            1,
            vec![MergedGeneration::new(shard1, 5)],
        )
        .unwrap();

        assert_eq!(indices.len(), 1);
        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        assert_eq!(details.merged_generations.len(), 1);
        assert_eq!(details.merged_generations[0].shard_id, shard1);
        assert_eq!(details.merged_generations[0].generation, 5);

        // Second update - updates existing shard
        update_mem_wal_index_merged_generations(
            &mut indices,
            2,
            vec![MergedGeneration::new(shard1, 10)],
        )
        .unwrap();

        assert_eq!(indices.len(), 1);
        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        assert_eq!(details.merged_generations.len(), 1);
        assert_eq!(details.merged_generations[0].generation, 10);

        // Third update - adds new shard
        update_mem_wal_index_merged_generations(
            &mut indices,
            3,
            vec![MergedGeneration::new(shard2, 3)],
        )
        .unwrap();

        assert_eq!(indices.len(), 1);
        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        assert_eq!(details.merged_generations.len(), 2);

        // Fourth update - lower generation should not update
        update_mem_wal_index_merged_generations(
            &mut indices,
            4,
            vec![MergedGeneration::new(shard1, 8)], // lower than 10
        )
        .unwrap();

        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        let r1_mg = details
            .merged_generations
            .iter()
            .find(|mg| mg.shard_id == shard1)
            .unwrap();
        assert_eq!(r1_mg.generation, 10); // Should still be 10
    }

    #[test]
    fn test_empty_merged_generations_noop() {
        let mut indices = Vec::new();

        // Empty update should be a no-op
        update_mem_wal_index_merged_generations(&mut indices, 1, vec![]).unwrap();

        assert!(indices.is_empty());
    }
}
