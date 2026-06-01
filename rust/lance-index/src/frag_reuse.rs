// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `Index`-trait adapter for the fragment-reuse system index.
//!
//! The data structures and table-format logic live in
//! [`lance_table::system_index::frag_reuse`]; this module re-exports them and
//! implements the local [`Index`] trait for [`FragReuseIndex`].

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use serde::Serialize;

pub use lance_table::system_index::frag_reuse::*;

use crate::{Index, IndexType};

#[derive(Serialize)]
struct FragReuseStatistics {
    num_versions: usize,
}

#[async_trait]
impl Index for FragReuseIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::not_supported_source(
            "FragReuseIndex is not a vector index".into(),
        ))
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = FragReuseStatistics {
            num_versions: self.details.versions.len(),
        };
        serde_json::to_value(stats).map_err(|e| {
            Error::internal(format!(
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
