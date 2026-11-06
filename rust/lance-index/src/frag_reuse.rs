// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `Index`-trait adapter for the fragment-reuse system index.
//!
//! The data structures and table-format logic live in
//! [`lance_table::system_index::frag_reuse`]; this module re-exports them and
//! provides a newtype wrapper that implements the [`Index`] trait.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;
use roaring::RoaringBitmap;
use serde::Serialize;

pub use lance_table::system_index::frag_reuse::*;

pub mod mapping_source;
pub mod row_map;
/// Fragment-level stable-partition mapping reader.
pub mod stable_partition;

use crate::{Index, IndexType};

/// Adapter for the compact runtime representation loaded from persisted FRI details.
#[doc(hidden)]
pub struct CompactFragReuseIndexHandle(pub Arc<CompactFragReuseIndex>);

#[derive(Serialize)]
struct FragReuseStatistics {
    num_versions: usize,
}

impl std::fmt::Debug for CompactFragReuseIndexHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CompactFragReuseIndexHandle")
            .field(&self.0)
            .finish()
    }
}

impl DeepSizeOf for CompactFragReuseIndexHandle {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

#[async_trait]
impl Index for CompactFragReuseIndexHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = FragReuseStatistics {
            num_versions: self.0.details.versions.len(),
        };
        serde_json::to_value(stats).map_err(|e| {
            lance_core::Error::internal(format!(
                "failed to serialize fragment reuse index statistics: {}",
                e
            ))
        })
    }

    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::FragmentReuse
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}
