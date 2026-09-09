// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared row-ID translation at asynchronous index loading boundaries.

use std::{collections::HashMap, sync::Arc};

use arrow_array::{RecordBatch, UInt64Array, cast::AsArray, types::UInt64Type};
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

    /// Prepare a synchronous remapper for the supplied batch before CPU-only decoding.
    ///
    /// In-memory mappings are returned directly. External mappings materialize only
    /// this batch; IDs outside the batch are left unchanged by the returned remapper.
    pub async fn prepare(&self, row_ids: &[u64]) -> Result<Arc<dyn RowIdRemapper>> {
        if let Self::InMemory(remapper) = self {
            return Ok(remapper.clone());
        }
        let mapped = self.remap_row_ids(row_ids).await?;
        Ok(Arc::new(PreparedRowIdRemapper(
            row_ids.iter().copied().zip(mapped).collect(),
        )))
    }

    /// Translate row IDs in input order, preserving duplicates and deleted positions.
    pub async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        match self {
            Self::InMemory(remapper) => Ok(row_ids
                .iter()
                .map(|id| remapper.remap_row_id(*id))
                .collect()),
            Self::External(remapper) => {
                let mapped = remapper.remap_row_ids(row_ids).await?;
                if mapped.len() != row_ids.len() {
                    return Err(Error::internal(format!(
                        "row-ID remapper returned {} results for {} inputs",
                        mapped.len(),
                        row_ids.len()
                    )));
                }
                Ok(mapped)
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
        let ids = batch
            .columns()
            .get(row_id_idx)
            .and_then(|array| array.as_primitive_opt::<UInt64Type>())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "row-ID column {row_id_idx} must exist and have type UInt64"
                ))
            })?;
        let inputs = ids.iter().flatten().collect::<Vec<_>>();
        self.prepare(&inputs)
            .await?
            .remap_row_ids_record_batch(batch, row_id_idx)
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
struct PreparedRowIdRemapper(HashMap<u64, Option<u64>>);

impl RowIdRemapper for PreparedRowIdRemapper {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        self.0.get(&row_id).copied().unwrap_or(Some(row_id))
    }

    fn remap_row_addrs_tree_map(&self, rows: &RowAddrTreeMap) -> RowAddrTreeMap {
        let mut result = rows.clone();
        for old in self.0.keys() {
            result.remove(*old);
        }
        for (old, new) in &self.0 {
            if rows.contains(*old)
                && let Some(new) = new
            {
                result.insert(*new);
            }
        }
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
    fn batch_preparation_preserves_positions_and_existing_decoder_semantics() {
        block_on(async {
            let external = Arc::new(ExternalMapping(AtomicUsize::new(0)));
            let remapping = RowIdRemapping::External(external.clone());
            let batch = record_batch!(
                ("value", Int32, [10, 30, 50, 11, 99]),
                ("ids", UInt64, [Some(1), Some(3), Some(5), Some(1), None]),
                ("extra", Boolean, [true, false, true, false, true])
            )
            .unwrap();
            let mapped = remapping
                .remap_row_ids_record_batch(batch, 1)
                .await
                .unwrap();
            assert_eq!(
                mapped,
                record_batch!(
                    ("value", Int32, [10, 50, 11]),
                    ("ids", UInt64, [Some(5), Some(1), Some(5)]),
                    ("extra", Boolean, [true, true, false])
                )
                .unwrap()
            );
            assert_eq!(external.0.load(Ordering::Relaxed), 1);

            let prepared = remapping.prepare(&[1, 3, 5]).await.unwrap();
            assert_eq!(prepared.remap_row_id(3), None);
            assert_eq!(prepared.remap_row_id(99), Some(99));
            let selected = RowAddrTreeMap::from_iter([1, 3, 5]);
            assert_eq!(
                prepared.remap_row_addrs_tree_map(&selected),
                RowAddrTreeMap::from_iter([1, 5])
            );
            assert_eq!(
                prepared.remap_row_ids_roaring_tree_map(&RoaringTreemap::from_iter([1, 3, 5])),
                RoaringTreemap::from_iter([1, 5])
            );
            assert_eq!(external.0.load(Ordering::Relaxed), 2);
        });
    }

    #[test]
    fn in_memory_preparation_keeps_the_existing_remapper() {
        block_on(async {
            let original: Arc<dyn RowIdRemapper> =
                Arc::new(PreparedRowIdRemapper(HashMap::from([(1, Some(5))])));
            let remapping = RowIdRemapping::InMemory(original.clone());
            let prepared = remapping.prepare(&[1]).await.unwrap();
            assert!(Arc::ptr_eq(&original, &prepared));
            let rows = RowAddrTreeMap::from_iter([1, 2]);
            assert_eq!(
                remapping.remap_row_addrs_tree_map(&rows).await.unwrap(),
                original.remap_row_addrs_tree_map(&rows)
            );
        });
    }
}
