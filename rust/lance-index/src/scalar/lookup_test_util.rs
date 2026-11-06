// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Helpers shared by the [`ScalarIndex::lookup`] tests of each index type.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::{ArrayRef, RecordBatch};
use futures::TryStreamExt;
use lance_select::Remapper;
use lance_select::remapper::{RemapperBacking, TranslateFuture};
use roaring::RoaringBitmap;

use crate::metrics::NoOpMetricsCollector;
use crate::scalar::{ScalarIndex, lookup_schema};

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

impl CompactedToFragmentOne {
    pub fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        (!self.deleted.contains(&row_id)).then(|| Self::moved(row_id))
    }

    pub fn into_remapper(self) -> Remapper {
        Remapper::from_backing(Arc::new(self))
    }
}

impl RemapperBacking for CompactedToFragmentOne {
    fn fragment_needs_translation(&self, fragment: u32) -> bool {
        fragment == 0
    }

    fn translate_fragments(&self, fragments: &RoaringBitmap) -> RoaringBitmap {
        let mut translated = fragments.clone();
        if translated.remove(0) {
            translated.insert(1);
        }
        translated
    }

    fn translate_in_place<'a>(&'a self, addrs: &'a mut [Option<u64>]) -> TranslateFuture<'a> {
        for slot in addrs.iter_mut() {
            *slot = slot.and_then(|addr| self.remap_row_id(addr));
        }
        Box::pin(std::future::ready(Ok(())))
    }
}
