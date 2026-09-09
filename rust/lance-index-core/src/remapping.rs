// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared row-ID translation at asynchronous index loading boundaries.

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

/// A shared remapper for both in-memory mappings and encodings requiring I/O.
///
/// Existing [`RowIdRemapper`] implementations keep their synchronous fast paths.
/// External encodings are awaited once per batch, never once per row.
///
/// ```
/// # use lance_index_core::remapping::RowIdRemapping;
/// # async fn example(remapping: &RowIdRemapping) -> lance_core::Result<()> {
/// let mapped = remapping.remap_row_ids(&[42, 43]).await?;
/// assert_eq!(mapped.len(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub enum RowIdRemapping {
    /// An existing synchronous remapper, including FRI V1.
    InMemory(Arc<dyn RowIdRemapper>),
    /// A mapping whose payload may need asynchronous reads.
    External(Arc<dyn BatchRowIdRemapper>),
}

impl RowIdRemapping {
    /// Require an in-memory mapping for an operation that cannot await translation.
    pub fn synchronous(&self) -> Result<&dyn RowIdRemapper> {
        match self {
            Self::InMemory(remapper) => Ok(remapper.as_ref()),
            Self::External(_) => Err(Error::not_supported(
                "this index maintenance operation does not support asynchronous row-ID remapping",
            )),
        }
    }

    /// Translate only the address column, leaving encoded columns in their original layout.
    ///
    /// The returned synchronous remapper removes tombstone slots during the
    /// consumer's layout-aware decoding. No per-row lookup table is retained.
    pub async fn remap_row_ids_preserving_layout(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<(RecordBatch, Arc<dyn RowIdRemapper>)> {
        if let Self::InMemory(remapper) = self {
            return Ok((batch, remapper.clone()));
        }
        let ids = batch
            .columns()
            .get(row_id_idx)
            .and_then(|array| array.as_primitive_opt::<UInt64Type>())
            .ok_or_else(|| {
                Error::invalid_input(format!("row-ID column {row_id_idx} must have type UInt64"))
            })?;
        let tombstone = lance_core::utils::address::RowAddress::TOMBSTONE_ROW;
        let mut translated = Vec::with_capacity(ids.len());
        for start in (0..ids.len()).step_by(64 * 1024) {
            let end = (start + 64 * 1024).min(ids.len());
            let inputs: Vec<_> = (start..end)
                .map(|position| {
                    if ids.is_null(position) {
                        tombstone
                    } else {
                        ids.value(position)
                    }
                })
                .collect();
            let mapped = self.remap_row_ids(&inputs).await?;
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

    /// Translate row IDs in input order, preserving duplicates and deleted positions.
    pub async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        match self {
            Self::InMemory(remapper) => Ok(row_ids
                .iter()
                .map(|id| remapper.remap_row_id(*id))
                .collect()),
            Self::External(remapper) => {
                let mut result = Vec::with_capacity(row_ids.len());
                for batch in row_ids.chunks(64 * 1024) {
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
        }
    }

    /// Translate a row-ID column and remove deleted rows from every column.
    pub async fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        if let Self::InMemory(remapper) = self {
            return remapper.remap_row_ids_record_batch(batch, row_id_idx);
        }
        let (batch, remapper) = self
            .remap_row_ids_preserving_layout(batch, row_id_idx)
            .await?;
        remapper.remap_row_ids_record_batch(batch, row_id_idx)
    }

    /// Translate an explicit physical row selection, dropping deleted addresses.
    pub async fn remap_row_addrs_tree_map(&self, rows: &RowAddrTreeMap) -> Result<RowAddrTreeMap> {
        if let Self::InMemory(remapper) = self {
            return Ok(remapper.remap_row_addrs_tree_map(rows));
        }
        let ids = rows.row_addrs().ok_or_else(|| Error::not_supported(
            "batch row-ID remapping requires explicit row addresses, not whole-fragment selections"
        ))?.map(u64::from);
        self.remap_iter(ids).await
    }

    /// Translate a bitmap of row IDs, dropping deleted rows.
    pub async fn remap_row_ids_roaring_tree_map(
        &self,
        rows: &RoaringTreemap,
    ) -> Result<RoaringTreemap> {
        if let Self::InMemory(remapper) = self {
            return Ok(remapper.remap_row_ids_roaring_tree_map(rows));
        }
        self.remap_iter(rows.iter()).await
    }

    async fn remap_iter<C: Default + Extend<u64>>(
        &self,
        mut ids: impl Iterator<Item = u64>,
    ) -> Result<C> {
        // Bitmap compression can hide millions of rows. Bound temporary translation buffers.
        const BATCH_SIZE: usize = 64 * 1024;
        let mut result = C::default();
        loop {
            let batch = ids.by_ref().take(BATCH_SIZE).collect::<Vec<_>>();
            if batch.is_empty() {
                break;
            }
            result.extend(self.remap_row_ids(&batch).await?.into_iter().flatten());
        }
        Ok(result)
    }
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
            let external = Arc::new(ExternalMapping(AtomicUsize::new(0)));
            let remapping = RowIdRemapping::External(external.clone());
            let batch = record_batch!(
                ("id", UInt64, [Some(1), Some(3), None, Some(5)]),
                ("value", Int32, [10, 30, 40, 50])
            )
            .unwrap();
            let translated = remapping
                .remap_row_ids_record_batch(batch, 0)
                .await
                .unwrap();
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
            let (batch, remapper) = remapping
                .remap_row_ids_preserving_layout(batch, 0)
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
    #[test]
    fn in_memory_keeps_the_existing_remapper() {
        block_on(async {
            let original: Arc<dyn RowIdRemapper> = Arc::new(TombstoneRowIdRemapper);
            let remapping = RowIdRemapping::InMemory(original.clone());
            let batch = record_batch!(("id", UInt64, [1])).unwrap();
            let (unchanged, remapper) = remapping
                .remap_row_ids_preserving_layout(batch.clone(), 0)
                .await
                .unwrap();
            assert_eq!(unchanged, batch);
            assert!(Arc::ptr_eq(&original, &remapper));
        });
    }
}
