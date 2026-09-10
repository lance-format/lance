// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Batch row-ID translation at asynchronous index loading boundaries.
//!
//! Legacy consumers keep calling [`RowIdRemapper`] directly; nothing in this
//! module participates in that path. The free helpers here serve the additive
//! entry points that load indices under a mapping whose payload may require
//! asynchronous reads, awaiting translation once per batch, never once per row.

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, UInt64Array, cast::AsArray, types::UInt64Type};
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_select::{RowAddrTreeMap, RowSetOps};
use roaring::RoaringTreemap;

use crate::scalar::RowIdRemapper;

/// Translates a bounded batch of row IDs, allowing encodings to read external data.
#[async_trait]
pub trait BatchRowIdRemapper: Send + Sync + std::fmt::Debug {
    /// Results correspond to input positions, including duplicates. `None` removes a row.
    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>>;
}

// Bitmap compression can hide millions of rows. Bound temporary translation buffers.
const BATCH_SIZE: usize = 64 * 1024;

tokio::task_local! {
    /// Armed by v0-lifecycle tests: any batch-remapping entry point reached
    /// within this scope fails, proving legacy traffic never crosses into the
    /// asynchronous translation code.
    pub static LEGACY_TRAFFIC_ONLY: ();
}

/// Fail when a legacy-only scope reaches a batch-remapping entry point.
///
/// Only debug builds (which include every test profile) perform the check;
/// release builds compile it away.
pub fn check_batch_remapping_entry() -> Result<()> {
    #[cfg(debug_assertions)]
    if LEGACY_TRAFFIC_ONLY.try_with(|_| ()).is_ok() {
        return Err(Error::internal(
            "legacy-only operation entered batch row-ID remapping",
        ));
    }
    Ok(())
}

/// Translate row IDs in input order, preserving duplicates and deleted positions.
pub async fn remap_row_ids_async(
    remapper: &dyn BatchRowIdRemapper,
    row_ids: &[u64],
) -> Result<Vec<Option<u64>>> {
    check_batch_remapping_entry()?;
    let mut result = Vec::with_capacity(row_ids.len());
    for batch in row_ids.chunks(BATCH_SIZE) {
        let mapped = remapper.remap_row_ids(batch).await?;
        if mapped.len() != batch.len() {
            return Err(Error::internal(format!(
                "row-ID remapper returned {} results for {} inputs",
                mapped.len(),
                batch.len()
            )));
        }
        result.extend(mapped);
    }
    Ok(result)
}

/// Translate only the address column, leaving encoded columns in their original layout.
///
/// The returned synchronous remapper removes tombstone slots during the
/// consumer's layout-aware decoding. No per-row lookup table is retained.
pub async fn remap_row_ids_preserving_layout_async(
    remapper: &dyn BatchRowIdRemapper,
    batch: RecordBatch,
    row_id_idx: usize,
) -> Result<(RecordBatch, Arc<dyn RowIdRemapper>)> {
    check_batch_remapping_entry()?;
    let ids = batch
        .columns()
        .get(row_id_idx)
        .and_then(|array| array.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            Error::invalid_input(format!("row-ID column {row_id_idx} must have type UInt64"))
        })?;
    let tombstone = lance_core::utils::address::RowAddress::TOMBSTONE_ROW;
    let mut translated = Vec::with_capacity(ids.len());
    for start in (0..ids.len()).step_by(BATCH_SIZE) {
        let end = (start + BATCH_SIZE).min(ids.len());
        let inputs: Vec<_> = (start..end)
            .map(|position| {
                if ids.is_null(position) {
                    tombstone
                } else {
                    ids.value(position)
                }
            })
            .collect();
        let mapped = remap_row_ids_async(remapper, &inputs).await?;
        translated.extend(mapped.into_iter().enumerate().map(|(offset, id)| {
            if ids.is_null(start + offset) {
                tombstone
            } else {
                id.unwrap_or(tombstone)
            }
        }));
    }
    let mut columns = batch.columns().to_vec();
    columns[row_id_idx] = Arc::new(UInt64Array::from(translated));
    let batch = RecordBatch::try_new(batch.schema(), columns)?;
    Ok((batch, Arc::new(TombstoneRowIdRemapper)))
}

/// Translate a row-ID column and remove deleted rows from every column.
pub async fn remap_record_batch_async(
    remapper: &dyn BatchRowIdRemapper,
    batch: RecordBatch,
    row_id_idx: usize,
) -> Result<RecordBatch> {
    check_batch_remapping_entry()?;
    let (batch, remapper) =
        remap_row_ids_preserving_layout_async(remapper, batch, row_id_idx).await?;
    remapper.remap_row_ids_record_batch(batch, row_id_idx)
}

/// Translate an explicit physical row selection, dropping deleted addresses.
pub async fn remap_row_addrs_tree_map_async(
    remapper: &dyn BatchRowIdRemapper,
    rows: &RowAddrTreeMap,
) -> Result<RowAddrTreeMap> {
    check_batch_remapping_entry()?;
    let ids = rows.row_addrs().ok_or_else(|| Error::not_supported(
        "batch row-ID remapping requires explicit row addresses, not whole-fragment selections"
    ))?.map(u64::from);
    remap_iter(remapper, ids).await
}

/// Translate a bitmap of row IDs, dropping deleted rows.
pub async fn remap_row_ids_roaring_tree_map_async(
    remapper: &dyn BatchRowIdRemapper,
    rows: &RoaringTreemap,
) -> Result<RoaringTreemap> {
    check_batch_remapping_entry()?;
    remap_iter(remapper, rows.iter()).await
}

async fn remap_iter<C: Default + Extend<u64>>(
    remapper: &dyn BatchRowIdRemapper,
    mut ids: impl Iterator<Item = u64>,
) -> Result<C> {
    let mut result = C::default();
    loop {
        let batch = ids.by_ref().take(BATCH_SIZE).collect::<Vec<_>>();
        if batch.is_empty() {
            break;
        }
        result.extend(
            remap_row_ids_async(remapper, &batch)
                .await?
                .into_iter()
                .flatten(),
        );
    }
    Ok(result)
}

#[derive(Debug)]
struct TombstoneRowIdRemapper;

impl RowIdRemapper for TombstoneRowIdRemapper {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        (row_id != lance_core::utils::address::RowAddress::TOMBSTONE_ROW).then_some(row_id)
    }

    fn remap_row_addrs_tree_map(&self, rows: &RowAddrTreeMap) -> RowAddrTreeMap {
        let mut result = rows.clone();
        result.remove(lance_core::utils::address::RowAddress::TOMBSTONE_ROW);
        result
    }

    fn remap_row_ids_roaring_tree_map(&self, rows: &RoaringTreemap) -> RoaringTreemap {
        rows.iter().filter_map(|id| self.remap_row_id(id)).collect()
    }

    fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        let ids = batch
            .columns()
            .get(row_id_idx)
            .and_then(|array| array.as_primitive_opt::<UInt64Type>())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "row-ID column {row_id_idx} must exist and have type UInt64"
                ))
            })?;
        let (positions, ids): (Vec<_>, Vec<_>) = ids
            .iter()
            .enumerate()
            .filter_map(|(position, id)| {
                id.and_then(|id| self.remap_row_id(id))
                    .map(|id| (position as u64, id))
            })
            .unzip();
        let positions = UInt64Array::from(positions);
        let mut columns = batch
            .columns()
            .iter()
            .map(|array| arrow_select::take::take(array, &positions, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        columns[row_id_idx] = Arc::new(UInt64Array::from(ids));
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::record_batch;
    use futures::executor::block_on;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct ExternalMapping(AtomicUsize);
    #[async_trait]
    impl BatchRowIdRemapper for ExternalMapping {
        async fn remap_row_ids(&self, ids: &[u64]) -> Result<Vec<Option<u64>>> {
            assert!(ids.len() <= 65536);
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(ids
                .iter()
                .map(|id| match *id {
                    1 => Some(5),
                    3 => None,
                    5 => Some(1),
                    other => Some(other),
                })
                .collect())
        }
    }
    #[test]
    fn external_remapping_bounds_batches_and_preserves_layout() {
        block_on(async {
            let external = ExternalMapping(AtomicUsize::new(0));
            let batch = record_batch!(
                ("id", UInt64, [Some(1), Some(3), None, Some(5)]),
                ("value", Int32, [10, 30, 40, 50])
            )
            .unwrap();
            let translated = remap_record_batch_async(&external, batch, 0).await.unwrap();
            assert_eq!(
                translated,
                record_batch!(
                    ("id", UInt64, [Some(5), Some(1)]),
                    ("value", Int32, [10, 50])
                )
                .unwrap()
            );
            let batch = RecordBatch::try_from_iter([(
                "id",
                Arc::new(UInt64Array::from(vec![1; 65537])) as arrow_array::ArrayRef,
            )])
            .unwrap();
            let (batch, remapper) = remap_row_ids_preserving_layout_async(&external, batch, 0)
                .await
                .unwrap();
            assert_eq!(batch.num_rows(), 65537);
            assert_eq!(external.0.load(Ordering::Relaxed), 3);
            assert_eq!(remapper.remap_row_id(5), Some(5));
            assert_eq!(
                remapper.remap_row_id(lance_core::utils::address::RowAddress::TOMBSTONE_ROW),
                None
            );
        });
    }
}
