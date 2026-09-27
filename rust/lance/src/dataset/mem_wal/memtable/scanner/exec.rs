// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DataFusion ExecutionPlan implementations for MemWAL read path.
//!
//! This module contains execution nodes for:
//! - `MemTableScanExec` - Full table scan with MVCC visibility
//! - `BTreeIndexExec` - BTree index queries
//! - `VectorIndexExec` - HNSW vector search
//! - `MemTableBruteForceVectorExec` - KNN over the active memtable without an HNSW
//! - `FtsIndexExec` - Full-text search

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{Fields, Schema};
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use lance_arrow::RecordBatchExt;

mod brute_force_vector;
mod btree;
mod dedup_scan;
mod fts;
mod scan;
mod vector;

use crate::dataset::mem_wal::scanner::exec::resolve_pk_indices;
use crate::dataset::mem_wal::write::BatchStore;

pub use brute_force_vector::MemTableBruteForceVectorExec;
pub use btree::BTreeIndexExec;
pub use dedup_scan::MemTableDedupScanExec;
pub use fts::{FtsIndexExec, SCORE_COLUMN};
pub use scan::{MemTableScanExec, ROW_ADDRESS_COLUMN};
pub use vector::VectorIndexExec;

/// Take `indices` out of `source_columns` and trim each to the shape
/// `output_schema` promises for that column name.
///
/// A projected struct leaf (`meta.a`) is served by taking the whole `meta`
/// column — the memtable stores columns whole — and narrowing it here.
/// `project_by_schema` recurses through structs, lists and maps and preserves
/// null buffers. Columns whose type already matches are passed through
/// untouched, so a projection with no nested paths costs nothing.
pub(super) fn take_projected_columns(
    source_columns: &[ArrayRef],
    source_fields: &Fields,
    indices: &[usize],
    output_schema: &Schema,
    num_rows: usize,
) -> DataFusionResult<Vec<ArrayRef>> {
    let mut out = Vec::with_capacity(indices.len());
    for &i in indices {
        let field = &source_fields[i];
        let column = source_columns[i].clone();
        let Ok(target) = output_schema.field_with_name(field.name()) else {
            out.push(column);
            continue;
        };
        if target.data_type() == field.data_type() {
            out.push(column);
            continue;
        }
        let src = RecordBatch::try_new_with_options(
            Arc::new(Schema::new(vec![field.clone()])),
            vec![column],
            &RecordBatchOptions::new().with_row_count(Some(num_rows)),
        )
        .map_err(DataFusionError::from)?;
        let narrowed = src
            .project_by_schema(&Schema::new(vec![Arc::new(target.clone())]))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        out.push(narrowed.column(0).clone());
    }
    Ok(out)
}

pub(super) fn newest_pk_positions(
    batch_store: &BatchStore,
    pk_columns: &[String],
    readable_count: usize,
    max_readable_row: u64,
) -> DataFusionResult<HashSet<u64>> {
    let mut newest: HashMap<Vec<ScalarValue>, u64> = HashMap::new();
    let mut current_row: u64 = 0;
    for (batch_position, stored_batch) in batch_store.iter().enumerate() {
        let n = stored_batch.num_rows;
        if n == 0 {
            continue;
        }
        if batch_position >= readable_count {
            current_row += n as u64;
            continue;
        }
        let pk_indices = resolve_pk_indices(&stored_batch.data, pk_columns)?;
        for row in 0..n {
            let pos = current_row + row as u64;
            if pos > max_readable_row {
                break;
            }
            let key = pk_key(&stored_batch.data, &pk_indices, row)?;
            newest.insert(key, pos);
        }
        current_row += n as u64;
    }
    Ok(newest.into_values().collect())
}

fn pk_key(
    batch: &RecordBatch,
    pk_indices: &[usize],
    row: usize,
) -> DataFusionResult<Vec<ScalarValue>> {
    pk_indices
        .iter()
        .map(|&col_idx| ScalarValue::try_from_array(batch.column(col_idx).as_ref(), row))
        .collect()
}
