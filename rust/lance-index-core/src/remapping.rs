// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Batch row-ID translation at asynchronous index loading boundaries.
//!
//! Legacy consumers keep calling [`RowIdRemapper`] directly; nothing in this
//! module participates in that path. The free helpers here serve the additive
//! entry points that load indices under a mapping whose payload may require
//! asynchronous reads, awaiting translation once per batch, never once per row.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use roaring::RoaringBitmap;

use arrow_array::{Array, RecordBatch, UInt64Array, cast::AsArray, types::UInt64Type};
use async_trait::async_trait;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{Error, Result};
use lance_select::{RowAddrTreeMap, RowSetOps};
use roaring::RoaringTreemap;

use crate::scalar::RowIdRemapper;

/// Translates a bounded batch of row IDs, allowing encodings to read external data.
#[async_trait]
pub trait BatchRowIdRemapper: Send + Sync + std::fmt::Debug {
    /// Results correspond to input positions, including duplicates. `None` removes a row.
    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>>;

    /// The physical row count of `fragment` in the address space this
    /// remapper translates from, when it knows it. The in-memory fallback of
    /// `remap_streaming` ([`materialize_remap`]) enumerates every address of
    /// the fragments an index holds; a fragment it cannot size makes the
    /// fallback decline. `None` (the default) means unknown.
    fn fragment_physical_rows(&self, _fragment: u32) -> Option<u64> {
        None
    }

    /// The memory the in-memory fallback may spend, temporary buffers and
    /// the materialized map together, before it declines with
    /// [`RemapUnavailable::OverBudget`].
    fn materialization_budget_bytes(&self) -> u64 {
        DEFAULT_MATERIALIZATION_BUDGET_BYTES
    }
}

/// The default [`BatchRowIdRemapper::materialization_budget_bytes`]: 256 MiB.
pub const DEFAULT_MATERIALIZATION_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Bytes charged per materialized map entry: a `HashMap<u64, Option<u64>>`
/// slot (8 + 16) plus control metadata and load-factor slack.
const MATERIALIZED_ENTRY_BYTES: u64 = 32;

/// Why an index could not be rewritten through a batch translator by the
/// in-memory fallback of `remap_streaming`, as opposed to an I/O or data
/// error: nothing is wrong with the index or the history, this build simply
/// cannot prove a complete mapping within its means. A maintenance job
/// leaves such a segment as it is (its files and the history it translates
/// through stay) and moves on; the reason travels inside
/// [`lance_core::Error::NotSupported`], see [`Self::from_error`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RemapUnavailable {
    /// The index does not report which fragments its files hold addresses
    /// for, so a complete mapping cannot be proven.
    StoredFragmentsUnknown,
    /// The translator cannot size a fragment the index holds.
    FragmentRowsUnknown { fragment: u32 },
    /// Materializing the mapping would exceed the translator's budget.
    OverBudget {
        estimated_bytes: u64,
        budget_bytes: u64,
    },
}

impl std::fmt::Display for RemapUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StoredFragmentsUnknown => write!(
                f,
                "the index does not report the fragments its files hold addresses for, so a \
                 complete in-memory mapping cannot be built for its legacy remap"
            ),
            Self::FragmentRowsUnknown { fragment } => write!(
                f,
                "the translator cannot size fragment {fragment}, which the index holds addresses \
                 for, so a complete in-memory mapping cannot be built for its legacy remap"
            ),
            Self::OverBudget {
                estimated_bytes,
                budget_bytes,
            } => write!(
                f,
                "materializing the mapping for the index's legacy remap needs about \
                 {estimated_bytes} bytes, above the {budget_bytes} byte budget"
            ),
        }
    }
}

impl std::error::Error for RemapUnavailable {}

impl RemapUnavailable {
    /// This reason as the error `remap_streaming` returns.
    pub fn into_error(self) -> Error {
        Error::not_supported_source(Box::new(self))
    }

    /// The reason carried by an error `remap_streaming` returned, if it
    /// declined for one; `None` for any other error (I/O, corrupt data, a
    /// translation failure), which a caller must not treat as a skip.
    pub fn from_error(error: &Error) -> Option<&Self> {
        match error {
            Error::NotSupported { source, .. } => source.downcast_ref::<Self>(),
            _ => None,
        }
    }
}

/// The complete in-memory mapping an index whose `remap` predates batch
/// translation needs: every address of every fragment in `stored_fragments`
/// translated through `remapper`, with a row the translator drops mapped
/// explicitly to `None`. The legacy `remap` treats an address missing from
/// its map as unchanged, so the fragments listed must be every fragment the
/// index's files may hold addresses for, retired ones included; a fragment
/// the index no longer claims but still stores would otherwise survive with
/// its stale addresses. An address the translator leaves as it is has the
/// same meaning either way and is not stored.
///
/// The cost is bounded before anything is allocated: the fragments' row
/// counts (from [`BatchRowIdRemapper::fragment_physical_rows`]) put an upper
/// bound on the map, and that plus the translation buffers must fit
/// [`BatchRowIdRemapper::materialization_budget_bytes`]. The fallback
/// declines with a [`RemapUnavailable`] when the fragments are unknown, a
/// fragment cannot be sized or the estimate is over budget; translation
/// errors propagate as they are.
pub async fn materialize_remap(
    remapper: &dyn BatchRowIdRemapper,
    stored_fragments: Option<&RoaringBitmap>,
) -> Result<RowAddrRemap> {
    let Some(fragments) = stored_fragments else {
        return Err(RemapUnavailable::StoredFragmentsUnknown.into_error());
    };
    let mut sized = Vec::with_capacity(fragments.len() as usize);
    let mut total_rows: u64 = 0;
    for fragment in fragments.iter() {
        let rows = remapper
            .fragment_physical_rows(fragment)
            .ok_or_else(|| RemapUnavailable::FragmentRowsUnknown { fragment }.into_error())?;
        total_rows = total_rows.saturating_add(rows);
        sized.push((fragment, rows));
    }
    let budget_bytes = remapper.materialization_budget_bytes();
    let estimated_bytes = total_rows
        .saturating_mul(MATERIALIZED_ENTRY_BYTES)
        .saturating_add((BATCH_SIZE as u64) * (8 + 16));
    if estimated_bytes > budget_bytes {
        return Err(RemapUnavailable::OverBudget {
            estimated_bytes,
            budget_bytes,
        }
        .into_error());
    }
    let mut map: HashMap<u64, Option<u64>> =
        HashMap::with_capacity(usize::try_from(total_rows).unwrap_or(usize::MAX));
    let mut slice: Vec<u64> = Vec::with_capacity(BATCH_SIZE.min(total_rows as usize));
    for (fragment, rows) in sized {
        let base = u64::from(fragment) << 32;
        let mut start = 0u64;
        while start < rows {
            let end = rows.min(start + BATCH_SIZE as u64);
            slice.clear();
            slice.extend((start..end).map(|offset| base | offset));
            let translated = remapper.remap_row_ids(&slice).await?;
            if translated.len() != slice.len() {
                return Err(Error::internal(format!(
                    "row-ID remapper returned {} results for {} inputs",
                    translated.len(),
                    slice.len()
                )));
            }
            for (address, result) in slice.iter().zip(translated) {
                if result != Some(*address) {
                    map.insert(*address, result);
                }
            }
            start = end;
        }
    }
    Ok(RowAddrRemap::direct(map))
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
/// direct map): every lookup is immediate. `Batch` is a translator whose
/// payload may need reads (a tagged history whose path holds stable-partition
/// hops). Index rewrites translate the addresses of one unit of work at a
/// time (a page, a partition, a spill batch) through [`Self::resolve`] or
/// [`Self::remap_row_addrs`], so a batch translator never has a map sized to
/// the source rows built for it. Both variants are shared handles, cheap to
/// clone into per-page tasks; [`Self::as_ref`] is the borrowed form the
/// legacy `remap` entry points reach without copying their map.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RowAddrTranslator {
    Sync(Arc<RowAddrRemap>),
    Batch(Arc<dyn BatchRowIdRemapper>),
}

impl RowAddrTranslator {
    /// A synchronous translator over an owned map.
    pub fn sync(remap: RowAddrRemap) -> Self {
        Self::Sync(Arc::new(remap))
    }

    /// A batch translator.
    pub fn batch(remapper: Arc<dyn BatchRowIdRemapper>) -> Self {
        Self::Batch(remapper)
    }

    /// The borrowed form.
    pub fn as_ref(&self) -> RowAddrTranslatorRef<'_> {
        match self {
            Self::Sync(remap) => RowAddrTranslatorRef::Sync(remap.as_ref()),
            Self::Batch(remapper) => RowAddrTranslatorRef::Batch(remapper.as_ref()),
        }
    }

    /// Whether the translator is known to move no address at all. A batch
    /// translator is never known to be empty.
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
}

/// A borrowed [`RowAddrTranslator`]: what an index rewrite works against.
/// The legacy `remap(&RowAddrRemap, ..)` entry points wrap their map in
/// [`Self::Sync`] and share one implementation with `remap_streaming`, so
/// neither copies the map nor delegates to the other.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum RowAddrTranslatorRef<'a> {
    Sync(&'a RowAddrRemap),
    Batch(&'a dyn BatchRowIdRemapper),
}

impl<'a> From<&'a RowAddrRemap> for RowAddrTranslatorRef<'a> {
    fn from(remap: &'a RowAddrRemap) -> Self {
        Self::Sync(remap)
    }
}

impl<'a> RowAddrTranslatorRef<'a> {
    /// Whether the translator is known to move no address at all. A batch
    /// translator is never known to be empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Sync(remap) => remap.is_empty(),
            Self::Batch(_) => false,
        }
    }

    /// Translate `addrs` in order, one result per input: `None` for a
    /// deleted row, otherwise the current address (unchanged when the
    /// translator does not touch it). At most `BATCH_SIZE` addresses are
    /// handed to a batch translator at a time.
    pub async fn remap_row_addrs(&self, addrs: &[u64]) -> Result<Vec<Option<u64>>> {
        match self {
            Self::Sync(remap) => Ok(addrs
                .iter()
                .map(|&addr| remap.get(addr).unwrap_or(Some(addr)))
                .collect()),
            Self::Batch(remapper) => remap_row_ids_async(*remapper, addrs).await,
        }
    }

    /// A synchronous map covering exactly `addrs`, so an existing synchronous
    /// remap step runs unchanged over one unit of work. `Sync` borrows its
    /// map; `Batch` translates `addrs` (deduplicated) and builds a map of
    /// that size, released with the returned value: memory is O(one unit),
    /// never O(source rows).
    pub async fn resolve(
        &self,
        addrs: impl IntoIterator<Item = u64>,
    ) -> Result<Cow<'a, RowAddrRemap>> {
        match self {
            Self::Sync(remap) => Ok(Cow::Borrowed(remap)),
            Self::Batch(remapper) => {
                let mut addrs: Vec<u64> = addrs.into_iter().collect();
                addrs.sort_unstable();
                addrs.dedup();
                let translated = remap_row_ids_async(*remapper, &addrs).await?;
                let map: HashMap<u64, Option<u64>> = addrs.into_iter().zip(translated).collect();
                Ok(Cow::Owned(RowAddrRemap::direct(map)))
            }
        }
    }
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
    use roaring::RoaringBitmap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Fragment 1 (rows move to fragment 5, offset 1 deleted), fragment 2
    /// (untouched) and fragment 3 (excluded: every row dropped).
    #[derive(Debug)]
    struct Sizing {
        calls: AtomicUsize,
        budget: u64,
        rows: HashMap<u32, u64>,
        fail: bool,
    }

    impl Sizing {
        fn new(rows: &[(u32, u64)], budget: u64) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                budget,
                rows: rows.iter().copied().collect(),
                fail: false,
            }
        }
    }

    #[async_trait]
    impl BatchRowIdRemapper for Sizing {
        async fn remap_row_ids(&self, ids: &[u64]) -> Result<Vec<Option<u64>>> {
            assert!(ids.len() <= BATCH_SIZE);
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                return Err(Error::io("the row map is gone"));
            }
            Ok(ids
                .iter()
                .map(|&address| {
                    let fragment = (address >> 32) as u32;
                    let offset = address & 0xffff_ffff;
                    match fragment {
                        1 if offset == 1 => None,
                        1 => Some((5u64 << 32) | offset),
                        2 => Some(address),
                        _ => None,
                    }
                })
                .collect())
        }

        fn fragment_physical_rows(&self, fragment: u32) -> Option<u64> {
            self.rows.get(&fragment).copied()
        }

        fn materialization_budget_bytes(&self) -> u64 {
            self.budget
        }
    }

    fn addr(fragment: u32, offset: u64) -> u64 {
        (u64::from(fragment) << 32) | offset
    }

    #[test]
    fn materialize_remap_maps_every_stored_address_explicitly() {
        let remapper = Sizing::new(&[(1, 4), (2, 3), (3, 2)], u64::MAX);
        let stored = RoaringBitmap::from_iter([1u32, 2, 3]);
        let remap = block_on(materialize_remap(&remapper, Some(&stored))).unwrap();
        // Moved rows, a deleted row and an excluded fragment are all explicit:
        // the legacy `remap` would keep any address it cannot find.
        assert_eq!(remap.get(addr(1, 0)), Some(Some(addr(5, 0))));
        assert_eq!(remap.get(addr(1, 1)), Some(None));
        assert_eq!(remap.get(addr(1, 3)), Some(Some(addr(5, 3))));
        assert_eq!(remap.get(addr(3, 0)), Some(None));
        assert_eq!(remap.get(addr(3, 1)), Some(None));
        // An address the translator leaves alone means "unchanged" either way.
        assert_eq!(remap.get(addr(2, 0)), None);
        // Nothing beyond the stored fragments' rows.
        assert_eq!(remap.get(addr(1, 4)), None);
        assert_eq!(remap.get(addr(4, 0)), None);
    }

    #[test]
    fn materialize_remap_declines_before_translating() {
        // Unknown stored fragments.
        let remapper = Sizing::new(&[(1, 4)], u64::MAX);
        let error = block_on(materialize_remap(&remapper, None)).unwrap_err();
        assert_eq!(
            RemapUnavailable::from_error(&error),
            Some(&RemapUnavailable::StoredFragmentsUnknown)
        );
        // A stored fragment the translator cannot size.
        let stored = RoaringBitmap::from_iter([1u32, 9]);
        let error = block_on(materialize_remap(&remapper, Some(&stored))).unwrap_err();
        assert_eq!(
            RemapUnavailable::from_error(&error),
            Some(&RemapUnavailable::FragmentRowsUnknown { fragment: 9 })
        );
        // Over budget: the estimate is made from the row counts alone.
        let tight = Sizing::new(&[(1, 4)], 1);
        let stored = RoaringBitmap::from_iter([1u32]);
        let error = block_on(materialize_remap(&tight, Some(&stored))).unwrap_err();
        match RemapUnavailable::from_error(&error) {
            Some(RemapUnavailable::OverBudget {
                budget_bytes: 1, ..
            }) => {}
            other => panic!("expected OverBudget, got {other:?}"),
        }
        assert!(matches!(error, Error::NotSupported { .. }));
        // None of these touched the translator.
        assert_eq!(remapper.calls.load(Ordering::Relaxed), 0);
        assert_eq!(tight.calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn materialize_remap_propagates_translation_errors() {
        let mut remapper = Sizing::new(&[(1, 4)], u64::MAX);
        remapper.fail = true;
        let stored = RoaringBitmap::from_iter([1u32]);
        let error = block_on(materialize_remap(&remapper, Some(&stored))).unwrap_err();
        // An I/O failure is an error, never a reason to skip.
        assert!(RemapUnavailable::from_error(&error).is_none());
        assert!(matches!(error, Error::IO { .. }), "{error}");
    }

    #[test]
    fn materialize_remap_translates_in_bounded_slices() {
        let rows = 2 * BATCH_SIZE as u64 + 1;
        let remapper = Sizing::new(&[(2, rows)], u64::MAX);
        let stored = RoaringBitmap::from_iter([2u32]);
        let remap = block_on(materialize_remap(&remapper, Some(&stored))).unwrap();
        assert_eq!(remapper.calls.load(Ordering::Relaxed), 3);
        // Every address was unchanged: nothing is stored for them.
        assert!(remap.is_empty());
    }

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
