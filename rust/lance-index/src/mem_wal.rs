// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `Index`-trait adapter for the MemWAL system index.
//!
//! The data structures and table-format logic live in
//! [`lance_table::system_index::mem_wal`]; this module re-exports them and
//! implements the local [`Index`] trait for [`MemWalIndex`].

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Error;
use roaring::RoaringBitmap;
use serde::Serialize;

pub use lance_table::system_index::mem_wal::*;

use crate::{Index, IndexType};

#[derive(Serialize)]
struct MemWalStatistics {
    num_shards: u32,
    num_merged_generations: usize,
    num_shard_specs: usize,
    num_maintained_indexes: usize,
    num_index_catchup_entries: usize,
}

#[async_trait]
impl Index for MemWalIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> lance_core::Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::not_supported_source(
            "MemWalIndex is not a vector index".into(),
        ))
    }

    fn statistics(&self) -> lance_core::Result<serde_json::Value> {
        let stats = MemWalStatistics {
            num_shards: self.details.num_shards,
            num_merged_generations: self.details.merged_generations.len(),
            num_shard_specs: self.details.sharding_specs.len(),
            num_maintained_indexes: self.details.maintained_indexes.len(),
            num_index_catchup_entries: self.details.index_catchup.len(),
        };
        serde_json::to_value(stats).map_err(|e| {
            Error::internal(format!(
                "failed to serialize MemWAL index statistics: {}",
                e
            ))
        })
    }

    async fn prewarm(&self) -> lance_core::Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::MemWal
    }

    async fn calculate_included_frags(&self) -> lance_core::Result<RoaringBitmap> {
        Ok(RoaringBitmap::new())
    }
}
