// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row-address translation for loading and rewriting indices.
//!
//! An index translates the addresses it stored through a [`Remapper`],
//! awaiting translation once per batch, never once per row. The free helpers
//! here apply one to the shapes indices store (bitmaps, record batches), and
//! [`RowAddrTranslator`] is the translation an index rewrite applies.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use roaring::RoaringBitmap;

use arrow_array::{Array, RecordBatch, UInt64Array, cast::AsArray, types::UInt64Type};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{Error, Result};
use lance_select::{RowAddrTreeMap, RowSetOps};
use roaring::RoaringTreemap;

use crate::scalar::RowIdRemapper;

pub use lance_select::Remapper;
pub use lance_select::remapper::{DEFAULT_MATERIALIZATION_BUDGET_BYTES, RemapUnavailable};

/// The translation of every row address in `addrs` through `remapper`,
/// without the ones that translate to `None`.
///
/// A posting can hold millions of addresses, so they are translated in
/// batches of at most `BATCH_SIZE` to bound the temporary buffer.
pub async fn translate_treemap(
    remapper: &Remapper,
    addrs: &RoaringTreemap,
) -> Result<RoaringTreemap> {
    let mut translated = RoaringTreemap::new();
    let mut iter = addrs.iter();
    let mut batch: Vec<Option<u64>> = Vec::with_capacity(BATCH_SIZE.min(addrs.len() as usize));
    loop {
        batch.clear();
        batch.extend(iter.by_ref().take(BATCH_SIZE).map(Some));
        if batch.is_empty() {
            return Ok(translated);
        }
        remapper.translate_in_place(&mut batch).await?;
        translated.extend(batch.iter().filter_map(|addr| *addr));
    }
}

/// Translate the row-address column at `addr_idx` of `batch` through
/// `remapper` without changing the batch's layout: a deleted row keeps its
/// slot, with the tombstone address. The returned synchronous remapper drops
/// tombstones, for consumers whose layout-aware decoding takes one.
#[doc(hidden)]
pub async fn translate_preserving_layout(
    remapper: &Remapper,
    batch: RecordBatch,
    addr_idx: usize,
) -> Result<(RecordBatch, Arc<dyn RowIdRemapper>)> {
    let addrs = batch
        .columns()
        .get(addr_idx)
        .and_then(|array| array.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "row address column {addr_idx} must have type UInt64"
            ))
        })?;
    let tombstone = lance_core::utils::address::RowAddress::TOMBSTONE_ROW;
    let mut translated = Vec::with_capacity(addrs.len());
    let mut slice: Vec<Option<u64>> = Vec::with_capacity(addrs.len().min(BATCH_SIZE));
    for start in (0..addrs.len()).step_by(BATCH_SIZE) {
        let end = (start + BATCH_SIZE).min(addrs.len());
        slice.clear();
        slice.extend(
            (start..end).map(|position| addrs.is_valid(position).then(|| addrs.value(position))),
        );
        remapper.translate_in_place(&mut slice).await?;
        translated.extend(slice.iter().map(|addr| addr.unwrap_or(tombstone)));
    }
    let mut columns = batch.columns().to_vec();
    columns[addr_idx] = Arc::new(UInt64Array::from(translated));
    let batch = RecordBatch::try_new(batch.schema(), columns)?;
    Ok((batch, Arc::new(TombstoneRowIdRemapper)))
}

/// Translate the row addresses in column `addr_idx` of `batch` through
/// `remapper`, as one batch, and drop the rows whose address translates to
/// `None`. A null address is dropped too: it names no row.
pub async fn translate_record_batch(
    remapper: &Remapper,
    batch: RecordBatch,
    addr_idx: usize,
) -> Result<RecordBatch> {
    let addrs = batch
        .columns()
        .get(addr_idx)
        .and_then(|array| array.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "row address column {addr_idx} must have type UInt64"
            ))
        })?;
    let mut translated: Vec<Option<u64>> = addrs.iter().collect();
    remapper.translate_in_place(&mut translated).await?;
    let kept =
        arrow_array::BooleanArray::from_iter(translated.iter().map(|addr| Some(addr.is_some())));
    let batch = arrow_select::filter::filter_record_batch(&batch, &kept)?;
    let mut columns = batch.columns().to_vec();
    columns[addr_idx] = Arc::new(UInt64Array::from_iter_values(
        translated.into_iter().flatten(),
    ));
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
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

/// The address translation an index rewrite applies, the argument of
/// `remap_streaming`.
///
/// `Sync` is a fully materialized map (a compaction's compact remap, or a
/// direct map): every lookup is immediate. `Remapper` is a translator whose
/// payload may need reads (a tagged history whose path holds stable-partition
/// hops). Index rewrites translate the addresses of one unit of work at a
/// time (a page, a partition, a spill batch) through [`Self::resolve`] or
/// [`Self::remap_row_addrs`], so a remapper never has a map sized to the
/// source rows built for it. Both variants are shared handles, cheap to
/// clone into per-page tasks; [`Self::as_ref`] is the borrowed form the
/// legacy `remap` entry points reach without copying their map.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RowAddrTranslator {
    Sync(Arc<RowAddrRemap>),
    /// A fragment reuse history's translation, bound from the plan
    /// `FragReuseHistory::compose` made. Whether it is ever held in memory
    /// is the history's decision ([`Remapper::materialize`]).
    Remapper(Remapper),
}

/// A remapper whose translation is already one in-memory map (a version 0
/// fragment reuse history's) becomes [`RowAddrTranslator::Sync`] over that
/// map, so the rewrite applies it exactly as the released version 0 remap
/// did, never through batch translation.
impl From<Remapper> for RowAddrTranslator {
    fn from(remapper: Remapper) -> Self {
        match remapper.row_addr_remap() {
            // One copy per rewrite. A version 0 remap is compact (sized by
            // rewrite groups and their changed-row bitmaps, not one entry per
            // row), so sharing it behind an Arc in CompactFragReuseIndex is
            // not worth widening that public type's internals.
            Some(remap) => Self::sync(remap.clone()),
            None => Self::Remapper(remapper),
        }
    }
}

impl RowAddrTranslator {
    /// A synchronous translator over an owned map.
    pub fn sync(remap: RowAddrRemap) -> Self {
        Self::Sync(Arc::new(remap))
    }

    /// The borrowed form.
    pub fn as_ref(&self) -> RowAddrTranslatorRef<'_> {
        match self {
            Self::Sync(remap) => RowAddrTranslatorRef::Sync(remap.as_ref()),
            Self::Remapper(remapper) => RowAddrTranslatorRef::Remapper(remapper),
        }
    }

    /// Whether the translator is known to move no address at all. A
    /// remapper is never known to be empty.
    pub fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    /// See [`RowAddrTranslatorRef::remap_row_addrs`].
    pub async fn remap_row_addrs(&self, addrs: &[u64]) -> Result<Vec<Option<u64>>> {
        self.as_ref().remap_row_addrs(addrs).await
    }

    /// See [`RowAddrTranslatorRef::resolve`].
    pub async fn resolve(
        &self,
        addrs: impl IntoIterator<Item = u64>,
    ) -> Result<Cow<'_, RowAddrRemap>> {
        self.as_ref().resolve(addrs).await
    }

    /// The complete in-memory map for an index whose rewrite takes one,
    /// covering every address in `stored_fragments`. A synchronous
    /// translator is its own map; any other is bounded and built by
    /// [`Remapper::materialize`], which declines with a
    /// [`RemapUnavailable`] when it cannot prove a complete map within
    /// budget.
    pub async fn materialize(
        &self,
        stored_fragments: Option<&RoaringBitmap>,
    ) -> Result<Cow<'_, RowAddrRemap>> {
        match self {
            Self::Sync(remap) => Ok(Cow::Borrowed(remap.as_ref())),
            Self::Remapper(remapper) => {
                Ok(Cow::Owned(remapper.materialize(stored_fragments).await?))
            }
        }
    }
}

/// A borrowed [`RowAddrTranslator`]: what an index rewrite works against.
/// The legacy `remap(&RowAddrRemap, ..)` entry points wrap their map in
/// [`Self::Sync`] and share one implementation with `remap_streaming`, so
/// neither copies the map nor delegates to the other.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum RowAddrTranslatorRef<'a> {
    Sync(&'a RowAddrRemap),
    Remapper(&'a Remapper),
}

impl<'a> From<&'a RowAddrRemap> for RowAddrTranslatorRef<'a> {
    fn from(remap: &'a RowAddrRemap) -> Self {
        Self::Sync(remap)
    }
}

impl<'a> RowAddrTranslatorRef<'a> {
    /// Whether the translator is known to move no address at all. A
    /// remapper is never known to be empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Sync(remap) => remap.is_empty(),
            Self::Remapper(_) => false,
        }
    }

    /// Translate `addrs` in order, one result per input: `None` for a
    /// deleted row, otherwise the current address (unchanged when the
    /// translator does not touch it). At most `BATCH_SIZE` addresses are
    /// handed to a remapper at a time.
    pub async fn remap_row_addrs(&self, addrs: &[u64]) -> Result<Vec<Option<u64>>> {
        match self {
            Self::Sync(remap) => Ok(addrs
                .iter()
                .map(|&addr| remap.get(addr).unwrap_or(Some(addr)))
                .collect()),
            Self::Remapper(remapper) => {
                check_batch_remapping_entry()?;
                let mut translated: Vec<Option<u64>> = addrs.iter().copied().map(Some).collect();
                for batch in translated.chunks_mut(BATCH_SIZE) {
                    remapper.translate_in_place(batch).await?;
                }
                Ok(translated)
            }
        }
    }

    /// A synchronous map covering exactly `addrs`, so an existing synchronous
    /// remap step runs unchanged over one unit of work. `Sync` borrows its
    /// map; `Remapper` translates `addrs` (deduplicated) and builds a map of
    /// that size, released with the returned value: memory is O(one unit),
    /// never O(source rows).
    pub async fn resolve(
        &self,
        addrs: impl IntoIterator<Item = u64>,
    ) -> Result<Cow<'a, RowAddrRemap>> {
        match self {
            Self::Sync(remap) => Ok(Cow::Borrowed(remap)),
            Self::Remapper(_) => {
                let mut addrs: Vec<u64> = addrs.into_iter().collect();
                addrs.sort_unstable();
                addrs.dedup();
                let translated = self.remap_row_addrs(&addrs).await?;
                let map: HashMap<u64, Option<u64>> = addrs.into_iter().zip(translated).collect();
                Ok(Cow::Owned(RowAddrRemap::direct(map)))
            }
        }
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
    use roaring::RoaringBitmap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Fragment 1 (rows move to fragment 5, offset 1 deleted), fragment 2
    /// (untouched) and every other fragment (every row dropped), counting
    /// the batches it translates.
    #[derive(Debug, Default)]
    struct Moves {
        calls: AtomicUsize,
    }

    impl lance_select::remapper::RemapperBacking for Moves {
        fn fragment_needs_translation(&self, _fragment: u32) -> bool {
            true
        }

        fn translate_fragments(&self, fragments: &RoaringBitmap) -> RoaringBitmap {
            fragments.clone()
        }

        fn translate_in_place<'a>(
            &'a self,
            addrs: &'a mut [Option<u64>],
        ) -> lance_select::remapper::TranslateFuture<'a> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            for slot in addrs.iter_mut() {
                *slot = slot.and_then(|address| {
                    let offset = address & 0xffff_ffff;
                    match (address >> 32) as u32 {
                        1 if offset == 1 => None,
                        1 => Some(addr(5, offset)),
                        2 => Some(address),
                        _ => None,
                    }
                });
            }
            Box::pin(std::future::ready(Ok(())))
        }
    }

    fn addr(fragment: u32, offset: u64) -> u64 {
        (u64::from(fragment) << 32) | offset
    }

    /// A [`Remapper`] following [`Moves`], and the calls it makes.
    fn moves_remapper() -> (Remapper, Arc<Moves>) {
        let moves = Arc::new(Moves::default());
        (Remapper::from_backing(moves.clone()), moves)
    }

    #[test]
    fn translate_preserving_layout_keeps_every_slot() {
        block_on(async {
            let (remapper, moves) = moves_remapper();
            let batch = record_batch!(
                (
                    "addr",
                    UInt64,
                    [Some(addr(1, 0)), None, Some(addr(1, 1)), Some(addr(2, 0))]
                ),
                ("value", Int32, [1, 2, 3, 4])
            )
            .unwrap();
            let (translated, tombstones) = translate_preserving_layout(&remapper, batch, 0)
                .await
                .unwrap();
            let tombstone = lance_core::utils::address::RowAddress::TOMBSTONE_ROW;
            assert_eq!(
                translated,
                record_batch!(
                    (
                        "addr",
                        UInt64,
                        [
                            Some(addr(5, 0)),
                            Some(tombstone),
                            Some(tombstone),
                            Some(addr(2, 0))
                        ]
                    ),
                    ("value", Int32, [1, 2, 3, 4])
                )
                .unwrap()
            );
            assert_eq!(tombstones.remap_row_id(tombstone), None);

            let rows = BATCH_SIZE + 1;
            let batch = RecordBatch::try_from_iter([(
                "addr",
                Arc::new(UInt64Array::from(vec![addr(1, 0); rows])) as arrow_array::ArrayRef,
            )])
            .unwrap();
            let before = moves.calls.load(Ordering::Relaxed);
            let (translated, _) = translate_preserving_layout(&remapper, batch, 0)
                .await
                .unwrap();
            assert_eq!(translated.num_rows(), rows);
            assert_eq!(moves.calls.load(Ordering::Relaxed) - before, 2);
        });
    }

    #[test]
    fn translate_record_batch_drops_deleted_and_null_rows() {
        block_on(async {
            let (remapper, _) = moves_remapper();
            let batch = record_batch!(
                (
                    "addr",
                    UInt64,
                    [Some(addr(1, 0)), None, Some(addr(1, 1)), Some(addr(2, 0))]
                ),
                ("value", Int32, [1, 2, 3, 4])
            )
            .unwrap();
            let translated = translate_record_batch(&remapper, batch.clone(), 0)
                .await
                .unwrap();
            assert_eq!(
                translated,
                record_batch!(
                    ("addr", UInt64, [addr(5, 0), addr(2, 0)]),
                    ("value", Int32, [1, 4])
                )
                .unwrap()
            );

            let error = translate_record_batch(&remapper, batch, 1)
                .await
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        });
    }

    #[test]
    fn translate_treemap_translates_in_bounded_batches() {
        block_on(async {
            let (remapper, moves) = moves_remapper();
            let addrs: RoaringTreemap = (0..=BATCH_SIZE as u64)
                .map(|offset| addr(1, offset))
                .collect();
            let translated = translate_treemap(&remapper, &addrs).await.unwrap();
            let expected: RoaringTreemap = (0..=BATCH_SIZE as u64)
                .filter(|&offset| offset != 1)
                .map(|offset| addr(5, offset))
                .collect();
            assert_eq!(translated, expected);
            assert_eq!(moves.calls.load(Ordering::Relaxed), 2);
        });
    }

    /// A backing that may or may not hold its translation as one map.
    #[derive(Debug)]
    struct MapBacking(Option<RowAddrRemap>);

    impl lance_select::remapper::RemapperBacking for MapBacking {
        fn fragment_needs_translation(&self, _fragment: u32) -> bool {
            true
        }

        fn translate_fragments(&self, fragments: &RoaringBitmap) -> RoaringBitmap {
            fragments.clone()
        }

        fn translate_in_place<'a>(
            &'a self,
            _addrs: &'a mut [Option<u64>],
        ) -> lance_select::remapper::TranslateFuture<'a> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn row_addr_remap(&self) -> Option<&RowAddrRemap> {
            self.0.as_ref()
        }
    }

    /// A backing moving fragment 1 to fragment 5 that records the size of
    /// every batch it is handed.
    #[derive(Debug, Default)]
    struct RecordingBatches(std::sync::Mutex<Vec<usize>>);

    impl lance_select::remapper::RemapperBacking for RecordingBatches {
        fn fragment_needs_translation(&self, fragment: u32) -> bool {
            fragment == 1
        }

        fn translate_fragments(&self, fragments: &RoaringBitmap) -> RoaringBitmap {
            fragments.clone()
        }

        fn translate_in_place<'a>(
            &'a self,
            addrs: &'a mut [Option<u64>],
        ) -> lance_select::remapper::TranslateFuture<'a> {
            self.0.lock().unwrap().push(addrs.len());
            for slot in addrs.iter_mut() {
                if let Some(address) = *slot
                    && address >> 32 == 1
                {
                    *slot = Some(addr(5, address & 0xffff_ffff));
                }
            }
            Box::pin(std::future::ready(Ok(())))
        }
    }

    /// Driven through the translator, a large request reaches the
    /// remapper in batches of at most `BATCH_SIZE` addresses with every
    /// position kept across batch boundaries, and `resolve` builds a map of
    /// exactly one unit of work.
    #[test]
    fn a_remapper_translator_bounds_batches_and_resolves_one_unit() {
        block_on(async {
            let backing = Arc::new(RecordingBatches::default());
            let translator = RowAddrTranslator::from(Remapper::from_backing(backing.clone()));
            assert!(matches!(translator, RowAddrTranslator::Remapper(_)));
            assert!(!translator.is_empty());

            let rows = 2 * BATCH_SIZE as u64 + 1;
            let input: Vec<u64> = (0..rows).map(|offset| addr(1, offset)).collect();
            let output = translator.remap_row_addrs(&input).await.unwrap();
            assert_eq!(output.len(), input.len());
            assert!(
                output
                    .iter()
                    .zip(0..rows)
                    .all(|(translated, offset)| *translated == Some(addr(5, offset)))
            );
            let batches = std::mem::take(&mut *backing.0.lock().unwrap());
            assert_eq!(batches, [BATCH_SIZE, BATCH_SIZE, 1]);

            let page = (10..20).map(|offset| addr(1, offset));
            let resolved = translator.resolve(page).await.unwrap();
            assert_eq!(resolved.get(addr(1, 15)), Some(Some(addr(5, 15))));
            assert_eq!(resolved.get(addr(1, 25)), None, "not part of the unit");
            assert_eq!(*backing.0.lock().unwrap(), [10]);
        });
    }

    #[test]
    fn an_in_memory_translation_rewrites_synchronously() {
        let remap = RowAddrRemap::direct(HashMap::from([(1, Some(5)), (3, None)]));
        let remapper = Remapper::from_backing(Arc::new(MapBacking(Some(remap.clone()))));
        let RowAddrTranslator::Sync(translated) = RowAddrTranslator::from(remapper) else {
            panic!("an in-memory translation is applied as its map");
        };
        assert_eq!(translated.as_ref(), &remap);

        let remapper = Remapper::from_backing(Arc::new(MapBacking(None)));
        assert!(matches!(
            RowAddrTranslator::from(remapper),
            RowAddrTranslator::Remapper(_)
        ));
    }
}
