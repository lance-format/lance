// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Helpers shared by the [`ScalarIndex::lookup`] tests of each index type.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use futures::TryStreamExt;
use lance_core::Result;
use lance_select::RowAddrTreeMap;
use roaring::RoaringTreemap;

use crate::metrics::NoOpMetricsCollector;
use crate::scalar::{RowIdRemapper, ScalarIndex, lookup_schema};

/// Every `(key position, row)` pair `lookup` streams, sorted, checking each
/// batch names its row column after the index's result domain.
pub async fn lookup_pairs(index: &dyn ScalarIndex, keys: ArrayRef) -> Vec<(u32, u64)> {
    let batches: Vec<RecordBatch> = index
        .lookup(keys, &NoOpMetricsCollector)
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut pairs = Vec::new();
    for batch in &batches {
        assert_eq!(
            batch.schema(),
            lookup_schema(index.results_are_row_addresses())
        );
        pairs.extend(
            batch
                .column(0)
                .as_primitive::<UInt32Type>()
                .values()
                .iter()
                .copied()
                .zip(
                    batch
                        .column(1)
                        .as_primitive::<UInt64Type>()
                        .values()
                        .iter()
                        .copied(),
                ),
        );
    }
    pairs.sort_unstable();
    pairs
}

/// Stands in for a deferred compaction: every row of fragment 0 moves to the
/// same offset in fragment 1, and the rows in `deleted` are dropped.
#[derive(Debug)]
pub struct CompactedToFragmentOne {
    pub deleted: HashSet<u64>,
}

impl CompactedToFragmentOne {
    pub fn moved(row: u64) -> u64 {
        (1 << 32) | row
    }
}

impl RowIdRemapper for CompactedToFragmentOne {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        (!self.deleted.contains(&row_id)).then(|| Self::moved(row_id))
    }

    fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        row_addrs
            .row_addrs()
            .unwrap()
            .filter_map(|addr| self.remap_row_id(u64::from(addr)))
            .collect()
    }

    fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap {
        row_ids
            .iter()
            .filter_map(|row_id| self.remap_row_id(row_id))
            .collect()
    }

    fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        let (positions, row_ids): (Vec<u64>, Vec<u64>) = batch
            .column(row_id_idx)
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .enumerate()
            .filter_map(|(position, &row_id)| {
                self.remap_row_id(row_id)
                    .map(|row_id| (position as u64, row_id))
            })
            .unzip();
        let positions = UInt64Array::from(positions);
        let mut columns = batch
            .columns()
            .iter()
            .map(|column| arrow_select::take::take(column, &positions, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        columns[row_id_idx] = Arc::new(UInt64Array::from(row_ids));
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    }
}
