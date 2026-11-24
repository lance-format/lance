// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! LSM Scanner - Unified scanner for LSM tree data
//!
//! This module provides scanners that read from multiple data sources
//! in an LSM tree architecture:
//! - Base table (compacted data)
//! - SSTables (persisted but not yet compacted)
//! - Active MemTable (in-memory buffer)
//!
//! The scanner handles deduplication by primary key, keeping the newest
//! version based on generation number and row address.
//!
//! ## Supported Query Types
//!
//! - **Scan**: Full table scan with deduplication
//! - **Point Lookup**: Primary key-based lookup with bloom filter optimization
//! - **Vector Search**: KNN search with staleness detection
//!
//! ## Example
//!
//! ```ignore
//! use lance::dataset::mem_wal::scanner::LsmScanner;
//!
//! let scanner = LsmScanner::new(base_table, shard_snapshots, vec!["pk".to_string()])
//!     .project(&["id", "name"])?
//!     .filter("id > 10")?
//!     .limit(Some(100), None)?;
//!
//! let stream = scanner.try_into_stream().await?;
//! ```

use arrow_array::Array;
use arrow_schema::Schema as ArrowSchema;
use datafusion::common::ToDFSchema;
use datafusion::prelude::{Expr, SessionContext};
use lance_core::{Error, Result};

mod block_list;
mod builder;
mod collector;
mod data_source;
pub mod exec;
mod fts_search;
mod generation_read;
mod planner;
mod point_lookup;
mod projection;
pub(crate) mod sstable_cache;
mod vector_search;

pub use block_list::write_pk_sidecar;
pub use builder::LsmScanner;
pub use collector::{
    ActiveMemTableRef, InMemoryMemTableRef, InMemoryMemTables, LsmDataSourceCollector,
};
pub use data_source::{FreshTierWatermark, LsmDataSource, LsmGeneration, ShardSnapshot, SsTable};
pub use fts_search::{LsmFtsSearchPlanner, SCORE_COLUMN};
pub use point_lookup::LsmPointLookupPlanner;
pub use projection::DISTANCE_COLUMN;
pub use sstable_cache::{DatasetCache, SsTableCache, SsTableWarmer};
pub use vector_search::LsmVectorSearchPlanner;

/// Parse a SQL filter expression against a MemWAL source schema.
pub fn parse_filter_expr(schema: &ArrowSchema, filter_expr: &str) -> Result<Expr> {
    let ctx = SessionContext::new();
    let df_schema = schema
        .clone()
        .to_dfschema()
        .map_err(|e| Error::invalid_input(format!("Failed to create DFSchema: {}", e)))?;
    ctx.parse_sql_expr(filter_expr, &df_schema)
        .map_err(|e| Error::invalid_input(format!("Failed to parse filter expression: {}", e)))
}

pub(crate) fn validate_overfetch_factor(factor: f64) -> Result<f64> {
    if !factor.is_finite() || factor < 1.0 {
        return Err(Error::invalid_input(format!(
            "overfetch_factor must be finite and at least 1.0, got {}",
            factor
        )));
    }
    Ok(factor)
}

/// The search selected by a MemWAL scanner builder's state setters.
pub(crate) enum SearchQuery<'a, V, F> {
    Scan,
    Vector(&'a V),
    FullText(&'a F),
}

/// The vector and full-text queries configured on a MemWAL scanner builder.
///
/// The public setters are state setters: a caller may call either one first,
/// and both may be present before planning. Keeping both slots together
/// makes the deferred mutual-exclusion rule and its single dispatch point
/// shared by the LSM and MemTable scanners.
#[derive(Clone, Debug)]
pub(crate) struct SearchState<V, F> {
    vector: Option<V>,
    full_text: Option<F>,
}

impl<V, F> Default for SearchState<V, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, F> SearchState<V, F> {
    pub(crate) fn new() -> Self {
        Self {
            vector: None,
            full_text: None,
        }
    }

    pub(crate) fn set_vector(&mut self, query: V) {
        self.vector = Some(query);
    }

    pub(crate) fn set_full_text(&mut self, query: F) {
        self.full_text = Some(query);
    }

    #[cfg(test)]
    pub(crate) fn vector(&self) -> Option<&V> {
        self.vector.as_ref()
    }

    pub(crate) fn vector_mut(&mut self) -> Option<&mut V> {
        self.vector.as_mut()
    }

    pub(crate) fn full_text_mut(&mut self) -> Option<&mut F> {
        self.full_text.as_mut()
    }

    pub(crate) fn has_search(&self) -> bool {
        self.vector.is_some() || self.full_text.is_some()
    }

    /// Resolve the configured searches into the one plan they select.
    ///
    /// The two queries are rejected together here, at plan time, rather than
    /// by the second setter, so both setters keep their existing state-setter
    /// behavior. `combined_error` carries each scanner's existing public error
    /// text.
    pub(crate) fn resolve(&self, combined_error: &str) -> Result<SearchQuery<'_, V, F>> {
        match (self.vector.as_ref(), self.full_text.as_ref()) {
            (Some(_), Some(_)) => Err(Error::invalid_input(combined_error.to_string())),
            (Some(vector), None) => Ok(SearchQuery::Vector(vector)),
            (None, Some(full_text)) => Ok(SearchQuery::FullText(full_text)),
            (None, None) => Ok(SearchQuery::Scan),
        }
    }
}

/// Validate the `nearest` setter arguments shared by both MemWAL scanners.
pub(crate) fn validate_nearest_query(query: &dyn Array, k: usize) -> Result<()> {
    if k == 0 {
        return Err(Error::invalid_input("k must be positive".to_string()));
    }
    if query.is_empty() {
        return Err(Error::invalid_input(
            "query vector must have non-zero length".to_string(),
        ));
    }
    Ok(())
}
