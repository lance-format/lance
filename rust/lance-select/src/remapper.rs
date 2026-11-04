// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Translation of the row addresses an index segment stored into the
//! addresses the rows have now.
//!
//! A rewrite (compaction) moves rows to new fragments. An index segment built
//! before it still stores the old addresses; a fragment reuse history records
//! the moves so the segment can be read without being rebuilt. [`Remapper`]
//! is the only view of that history an index sees: it answers which
//! fragments need translation, and translates addresses in batches.
//!
//! Translation is always asynchronous, because some history formats read
//! their mappings lazily. A format whose mappings are already in memory
//! returns futures that are complete on first poll. Indices never learn
//! which format produced their remapper.

use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use crate::mask::{RowAddrSelection, RowAddrTreeMap};

/// The future [`RemapperBacking::translate_in_place`] returns.
#[doc(hidden)]
pub type TranslateFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

/// What produces a [`Remapper`]: a fragment reuse history format (in
/// `lance-table`), or a scripted test remapper. Not an extension point; the
/// formats own the semantics, and indices only see [`Remapper`].
#[doc(hidden)]
pub trait RemapperBacking: Send + Sync + Debug {
    fn fragment_needs_translation(&self, fragment: u32) -> bool;

    fn translate_fragments(&self, fragments: &RoaringBitmap) -> RoaringBitmap;

    fn translate_in_place<'a>(&'a self, addrs: &'a mut [Option<u64>]) -> TranslateFuture<'a>;

    /// The physical row count of `fragment` in the address space translated
    /// from, when the backing knows it. [`Remapper::materialize`] declines
    /// for a fragment that needs translation and cannot be sized.
    fn fragment_physical_rows(&self, _fragment: u32) -> Option<u64> {
        None
    }

    /// The memory [`Remapper::materialize`] may spend, temporary buffers and
    /// the map together.
    fn materialization_budget_bytes(&self) -> u64 {
        DEFAULT_MATERIALIZATION_BUDGET_BYTES
    }
}

/// Translates the row addresses one index segment stores into the addresses
/// the rows have in the snapshot being read.
///
/// Cheap to clone. Fragments the remapper does not need to translate keep
/// their addresses; an address whose row was deleted, or that the segment no
/// longer answers for, translates to `None`.
///
/// An index is handed a remapper when it is loaded (see
/// `lance_index::scalar::registry::ScalarIndexPlugin::load_index`) and
/// translates the addresses it reads in batches:
///
/// ```
/// # use lance_core::Result;
/// # use lance_select::Remapper;
/// /// The current addresses of the rows a segment stored at `stored`,
/// /// without the rows a rewrite deleted.
/// async fn current_addresses(remapper: &Remapper, stored: &[u64]) -> Result<Vec<u64>> {
///     let mut addresses: Vec<Option<u64>> = stored.iter().copied().map(Some).collect();
///     remapper.translate_in_place(&mut addresses).await?;
///     Ok(addresses.into_iter().flatten().collect())
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Remapper(Arc<dyn RemapperBacking>);

impl Remapper {
    #[doc(hidden)]
    pub fn from_backing(backing: Arc<dyn RemapperBacking>) -> Self {
        Self(backing)
    }

    /// Whether addresses in `fragment` may change under translation. When
    /// this is false, every address in `fragment` translates to itself, so
    /// an index can skip translating them.
    pub fn fragment_needs_translation(&self, fragment: u32) -> bool {
        self.0.fragment_needs_translation(fragment)
    }

    /// The fragments that rows stored in `fragments` may have after
    /// translation: fragments that need no translation are kept, and the
    /// others are replaced by the fragments their rows moved to.
    pub fn translate_fragments(&self, fragments: &RoaringBitmap) -> RoaringBitmap {
        self.0.translate_fragments(fragments)
    }

    /// Translate `addrs` in place. A `None` stays `None`; a `Some` becomes
    /// the row's current address, or `None` when the row is gone. The whole
    /// slice is one batch: pass as many addresses as are at hand rather than
    /// one at a time, since a lazily read mapping is read once per batch.
    pub async fn translate_in_place(&self, addrs: &mut [Option<u64>]) -> Result<()> {
        self.0.translate_in_place(addrs).await
    }

    /// The translation of every address in `addrs`, without the ones that
    /// translate to `None`.
    ///
    /// Fails with `InvalidInput` when `addrs` selects a whole fragment that
    /// needs translation: its addresses cannot be enumerated.
    pub async fn translate_set(&self, addrs: &RowAddrTreeMap) -> Result<RowAddrTreeMap> {
        let mut translated = RowAddrTreeMap::new();
        let mut pending = Vec::new();
        for (&fragment, selection) in addrs.iter() {
            if !self.fragment_needs_translation(fragment) {
                match selection {
                    RowAddrSelection::Full => translated.insert_fragment(fragment),
                    RowAddrSelection::Partial(bitmap) => {
                        translated.insert_bitmap(fragment, bitmap.clone())
                    }
                }
                continue;
            }
            let RowAddrSelection::Partial(bitmap) = selection else {
                return Err(Error::invalid_input(format!(
                    "cannot translate a selection of all rows of fragment {fragment}: the \
                     addresses of a fragment that needs translation must be listed"
                )));
            };
            let base = u64::from(fragment) << 32;
            pending.extend(bitmap.iter().map(|offset| Some(base | u64::from(offset))));
        }
        if !pending.is_empty() {
            self.translate_in_place(&mut pending).await?;
        }
        for addr in pending.into_iter().flatten() {
            translated.insert(addr);
        }
        Ok(translated)
    }

    /// The complete in-memory map of every address in `stored_fragments`
    /// that translation changes, a row it drops mapped explicitly to `None`,
    /// for an index whose rewrite takes a [`RowAddrRemap`] (where a missing
    /// address means "unchanged"). `stored_fragments` must list every
    /// fragment the index's files may hold addresses for, retired ones
    /// included.
    ///
    /// Whether this is affordable is the remapper's decision, not the
    /// caller's: the fragment reuse history that produced it knows the row
    /// counts of the fragments it translates, and bounds the map. The cost is
    /// checked before anything is allocated, and the call declines with a
    /// [`RemapUnavailable`] (inside [`Error::NotSupported`]) when the
    /// fragments are unknown, one that needs translation cannot be sized, or
    /// the map would exceed the budget. Translation errors propagate as they
    /// are.
    pub async fn materialize(
        &self,
        stored_fragments: Option<&RoaringBitmap>,
    ) -> Result<RowAddrRemap> {
        let Some(fragments) = stored_fragments else {
            return Err(RemapUnavailable::StoredFragmentsUnknown.into_error());
        };
        let mut sized = Vec::with_capacity(fragments.len() as usize);
        let mut total_rows: u64 = 0;
        for fragment in fragments
            .iter()
            .filter(|&fragment| self.fragment_needs_translation(fragment))
        {
            let rows = self
                .0
                .fragment_physical_rows(fragment)
                .ok_or_else(|| RemapUnavailable::FragmentRowsUnknown { fragment }.into_error())?;
            total_rows = total_rows.saturating_add(rows);
            sized.push((fragment, rows));
        }
        let budget_bytes = self.0.materialization_budget_bytes();
        let estimated_bytes =
            hash_map_bytes(total_rows).saturating_add((MATERIALIZE_BATCH_SIZE as u64) * (8 + 16));
        if estimated_bytes > budget_bytes {
            return Err(RemapUnavailable::OverBudget {
                estimated_bytes,
                budget_bytes,
            }
            .into_error());
        }
        let mut map: HashMap<u64, Option<u64>> =
            HashMap::with_capacity(usize::try_from(total_rows).unwrap_or(usize::MAX));
        let mut slice: Vec<Option<u64>> =
            Vec::with_capacity(MATERIALIZE_BATCH_SIZE.min(total_rows as usize));
        for (fragment, rows) in sized {
            let base = u64::from(fragment) << 32;
            let mut start = 0u64;
            while start < rows {
                let end = rows.min(start + MATERIALIZE_BATCH_SIZE as u64);
                slice.clear();
                slice.extend((start..end).map(|offset| Some(base | offset)));
                self.translate_in_place(&mut slice).await?;
                for (offset, result) in (start..end).zip(&slice) {
                    let address = base | offset;
                    if *result != Some(address) {
                        map.insert(address, *result);
                    }
                }
                start = end;
            }
        }
        Ok(RowAddrRemap::direct(map))
    }
}

/// Addresses [`Remapper::materialize`] translates at a time. Bitmap
/// compression can hide millions of rows, so the buffer is bounded.
const MATERIALIZE_BATCH_SIZE: usize = 64 * 1024;

/// The default [`RemapperBacking::materialization_budget_bytes`]: 256 MiB.
pub const DEFAULT_MATERIALIZATION_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Bytes hashbrown allocates for the `HashMap<u64, Option<u64>>` that
/// `HashMap::with_capacity(entries)` reserves.
///
/// hashbrown's `capacity_to_buckets` rounds the capacity up to
/// `next_power_of_two(ceil(entries * 8 / 7))` buckets (4 below four entries, 8
/// below eight) and lays the table out as one slot of
/// `size_of::<(u64, Option<u64>)>()` (24) bytes plus one control byte per
/// bucket, followed by one group width (16) of trailing control bytes. The
/// rounding is what a flat per-entry charge misses: 7,350,000 entries adjust
/// to 8,400,000 and round to 16,777,216 buckets, 419,430,416 bytes, where 32
/// bytes per entry would have charged 235,200,000.
#[doc(hidden)]
pub fn hash_map_bytes(entries: u64) -> u64 {
    const SLOT_BYTES: u64 = std::mem::size_of::<(u64, Option<u64>)>() as u64;
    const CTRL_BYTES_PER_BUCKET: u64 = 1;
    const GROUP_WIDTH: u64 = 16;
    let buckets = if entries < 4 {
        4
    } else if entries < 8 {
        8
    } else {
        entries
            .saturating_mul(8)
            .div_ceil(7)
            .checked_next_power_of_two()
            .unwrap_or(u64::MAX)
    };
    buckets
        .saturating_mul(SLOT_BYTES + CTRL_BYTES_PER_BUCKET)
        .saturating_add(GROUP_WIDTH)
}

/// Why an index could not be rewritten through a remapper by the in-memory
/// fallback ([`Remapper::materialize`]), as opposed to an I/O or data error:
/// nothing is wrong with the index or the history, this build simply cannot
/// prove a complete mapping within its means. A maintenance job leaves such a
/// segment as it is (its files and the history it translates through stay)
/// and moves on; the reason travels inside [`Error::NotSupported`], see
/// [`Self::from_error`].
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
    /// This reason as the error an index rewrite returns.
    pub fn into_error(self) -> Error {
        Error::not_supported_source(Box::new(self))
    }

    /// The reason carried by an error an index rewrite returned, if it
    /// declined for one; `None` for any other error (I/O, corrupt data, a
    /// translation failure), which a caller must not treat as a skip.
    pub fn from_error(error: &Error) -> Option<&Self> {
        match error {
            Error::NotSupported { source, .. } => source.downcast_ref::<Self>(),
            _ => None,
        }
    }
}

/// A [`Remapper`] that follows a script instead of a fragment reuse history,
/// for testing indices without building one.
///
/// Every fragment holding a scripted address is rewritten: each of its
/// addresses must be scripted, as moved or deleted, and translating one that
/// is not fails. Other fragments are untouched. The script counts the
/// batches it translates, so a test can check that an index translates in
/// batches rather than once per row.
#[cfg(any(test, feature = "test-util"))]
#[derive(Debug, Default)]
pub struct ScriptedRemapper {
    moves: std::collections::HashMap<u64, Option<u64>>,
    rewritten: RoaringBitmap,
    fragment_rows: std::collections::HashMap<u32, u64>,
    budget_bytes: Option<u64>,
    batches: std::sync::atomic::AtomicUsize,
    addresses: std::sync::atomic::AtomicUsize,
}

#[cfg(any(test, feature = "test-util"))]
impl ScriptedRemapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// The row at `from` now lives at `to`.
    pub fn moved(mut self, from: u64, to: u64) -> Self {
        self.script(from, Some(to));
        self
    }

    /// The row at `addr` was deleted.
    pub fn deleted(mut self, addr: u64) -> Self {
        self.script(addr, None);
        self
    }

    /// Fragment `fragment` holds `rows` physical rows, so
    /// [`Remapper::materialize`] can enumerate it.
    pub fn sized(mut self, fragment: u32, rows: u64) -> Self {
        self.fragment_rows.insert(fragment, rows);
        self
    }

    /// [`Remapper::materialize`] may spend at most `bytes`.
    pub fn with_budget(mut self, bytes: u64) -> Self {
        self.budget_bytes = Some(bytes);
        self
    }

    fn script(&mut self, addr: u64, translated: Option<u64>) {
        self.rewritten.insert((addr >> 32) as u32);
        self.moves.insert(addr, translated);
    }

    /// The remapper following this script; its calls are counted here.
    pub fn into_remapper(self) -> (Remapper, Arc<Self>) {
        let script = Arc::new(self);
        (Remapper::from_backing(script.clone()), script)
    }

    /// How many batches [`Remapper::translate_in_place`] was called with.
    pub fn batches(&self) -> usize {
        self.batches.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// How many addresses were translated, over every batch.
    pub fn addresses(&self) -> usize {
        self.addresses.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(any(test, feature = "test-util"))]
impl RemapperBacking for ScriptedRemapper {
    fn fragment_needs_translation(&self, fragment: u32) -> bool {
        self.rewritten.contains(fragment)
    }

    fn translate_fragments(&self, fragments: &RoaringBitmap) -> RoaringBitmap {
        let mut translated = fragments - &self.rewritten;
        translated.extend(self.moves.iter().filter_map(|(&from, &to)| {
            to.filter(|_| fragments.contains((from >> 32) as u32))
                .map(|to| (to >> 32) as u32)
        }));
        translated
    }

    fn translate_in_place<'a>(&'a self, addrs: &'a mut [Option<u64>]) -> TranslateFuture<'a> {
        use std::sync::atomic::Ordering;
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.addresses.fetch_add(addrs.len(), Ordering::Relaxed);
        let result = addrs.iter_mut().try_for_each(|slot| {
            let Some(addr) = *slot else {
                return Ok(());
            };
            if !self.rewritten.contains((addr >> 32) as u32) {
                return Ok(());
            }
            *slot = *self.moves.get(&addr).ok_or_else(|| {
                Error::invalid_input(format!(
                    "the script rewrites fragment {} but not its address {addr:#x}",
                    addr >> 32
                ))
            })?;
            Ok(())
        });
        Box::pin(std::future::ready(result))
    }

    fn fragment_physical_rows(&self, fragment: u32) -> Option<u64> {
        self.fragment_rows.get(&fragment).copied()
    }

    fn materialization_budget_bytes(&self) -> u64 {
        self.budget_bytes
            .unwrap_or(DEFAULT_MATERIALIZATION_BUDGET_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address(fragment: u64, offset: u64) -> u64 {
        (fragment << 32) | offset
    }

    /// Fragment 1 was rewritten into fragment 3; fragment 2 is untouched.
    fn script() -> (Remapper, Arc<ScriptedRemapper>) {
        ScriptedRemapper::new()
            .moved(address(1, 0), address(3, 1))
            .deleted(address(1, 1))
            .moved(address(1, 2), address(3, 0))
            .into_remapper()
    }

    #[tokio::test]
    async fn translates_moved_deleted_and_untouched_rows_in_one_batch() {
        let (remapper, script) = script();
        let mut addrs = vec![
            Some(address(1, 0)),
            Some(address(1, 1)),
            Some(address(2, 7)),
            None,
            Some(address(1, 2)),
        ];
        remapper.translate_in_place(&mut addrs).await.unwrap();
        assert_eq!(
            addrs,
            vec![
                Some(address(3, 1)),
                None,
                Some(address(2, 7)),
                None,
                Some(address(3, 0)),
            ]
        );
        assert_eq!(script.batches(), 1);
        assert_eq!(script.addresses(), 5);
    }

    #[tokio::test]
    async fn translate_set_keeps_untouched_fragments_whole() {
        let (remapper, script) = script();
        let mut addrs = RowAddrTreeMap::new();
        addrs.insert(address(1, 0));
        addrs.insert(address(1, 1));
        addrs.insert_fragment(2);
        let translated = remapper.translate_set(&addrs).await.unwrap();
        let mut expected = RowAddrTreeMap::new();
        expected.insert(address(3, 1));
        expected.insert_fragment(2);
        assert_eq!(translated, expected);

        let untouched = RowAddrTreeMap::from_iter([address(2, 7)]);
        let before = script.batches();
        assert_eq!(remapper.translate_set(&untouched).await.unwrap(), untouched);
        assert_eq!(script.batches(), before, "nothing needed translation");

        let mut whole_rewritten = RowAddrTreeMap::new();
        whole_rewritten.insert_fragment(1);
        let error = remapper.translate_set(&whole_rewritten).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("fragment 1"), "{error}");
    }

    #[tokio::test]
    async fn translate_set_of_nothing_translates_nothing() {
        let (remapper, script) = script();
        let translated = remapper
            .translate_set(&RowAddrTreeMap::new())
            .await
            .unwrap();
        assert_eq!(translated, RowAddrTreeMap::new());
        assert_eq!(script.batches(), 0);
    }

    /// Two stored addresses that now name the same row (a sibling's rewrite
    /// merged them) come back as that one address.
    #[tokio::test]
    async fn translate_set_merges_addresses_that_collapse() {
        let (remapper, _) = ScriptedRemapper::new()
            .moved(address(1, 0), address(3, 4))
            .moved(address(2, 0), address(3, 4))
            .into_remapper();
        let addrs = RowAddrTreeMap::from_iter([address(1, 0), address(2, 0)]);
        assert_eq!(
            remapper.translate_set(&addrs).await.unwrap(),
            RowAddrTreeMap::from_iter([address(3, 4)])
        );
    }

    #[test]
    fn lineage_questions_need_no_translation() {
        let (remapper, script) = script();
        assert!(remapper.fragment_needs_translation(1));
        assert!(!remapper.fragment_needs_translation(2));
        assert_eq!(
            remapper.translate_fragments(&RoaringBitmap::from_iter([1, 2])),
            RoaringBitmap::from_iter([2, 3])
        );
        assert_eq!(script.batches(), 0);
    }

    /// Fragment 1's four rows move to fragment 5 (offset 1 deleted), fragment
    /// 3's two rows are dropped, fragment 2 is untouched.
    fn sized_script(budget: u64) -> (Remapper, Arc<ScriptedRemapper>) {
        ScriptedRemapper::new()
            .moved(address(1, 0), address(5, 0))
            .deleted(address(1, 1))
            .moved(address(1, 2), address(5, 2))
            .moved(address(1, 3), address(5, 3))
            .deleted(address(3, 0))
            .deleted(address(3, 1))
            .sized(1, 4)
            .sized(3, 2)
            .with_budget(budget)
            .into_remapper()
    }

    #[tokio::test]
    async fn materialize_maps_every_changed_address_explicitly() {
        let (remapper, _) = sized_script(u64::MAX);
        let stored = RoaringBitmap::from_iter([1u32, 2, 3]);
        let remap = remapper.materialize(Some(&stored)).await.unwrap();
        // Moved rows, a deleted row and a dropped fragment are all explicit:
        // the map's consumer keeps any address it cannot find.
        assert_eq!(remap.get(address(1, 0)), Some(Some(address(5, 0))));
        assert_eq!(remap.get(address(1, 1)), Some(None));
        assert_eq!(remap.get(address(1, 3)), Some(Some(address(5, 3))));
        assert_eq!(remap.get(address(3, 0)), Some(None));
        assert_eq!(remap.get(address(3, 1)), Some(None));
        // An untouched fragment needs no size and nothing is stored for it.
        assert_eq!(remap.get(address(2, 0)), None);
        assert_eq!(remap.get(address(1, 4)), None);
    }

    #[tokio::test]
    async fn materialize_declines_before_translating() {
        let (remapper, script) = sized_script(u64::MAX);
        let error = remapper.materialize(None).await.unwrap_err();
        assert_eq!(
            RemapUnavailable::from_error(&error),
            Some(&RemapUnavailable::StoredFragmentsUnknown)
        );
        let (unsizable, unsized_script) = ScriptedRemapper::new()
            .deleted(address(9, 0))
            .into_remapper();
        let error = unsizable
            .materialize(Some(&RoaringBitmap::from_iter([9u32])))
            .await
            .unwrap_err();
        assert_eq!(
            RemapUnavailable::from_error(&error),
            Some(&RemapUnavailable::FragmentRowsUnknown { fragment: 9 })
        );
        let (tight, tight_script) = sized_script(1);
        let error = tight
            .materialize(Some(&RoaringBitmap::from_iter([1u32])))
            .await
            .unwrap_err();
        assert!(matches!(
            RemapUnavailable::from_error(&error),
            Some(RemapUnavailable::OverBudget {
                budget_bytes: 1,
                ..
            })
        ));
        assert!(matches!(error, Error::NotSupported { .. }));
        assert_eq!(script.batches(), 0);
        assert_eq!(unsized_script.batches(), 0);
        assert_eq!(tight_script.batches(), 0);
    }

    #[tokio::test]
    async fn materialize_propagates_translation_errors() {
        // Fragment 1 is rewritten but only offset 0 is scripted.
        let (remapper, _) = ScriptedRemapper::new()
            .moved(address(1, 0), address(2, 0))
            .sized(1, 4)
            .into_remapper();
        let error = remapper
            .materialize(Some(&RoaringBitmap::from_iter([1u32])))
            .await
            .unwrap_err();
        assert!(RemapUnavailable::from_error(&error).is_none());
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }

    #[tokio::test]
    async fn materialize_translates_in_bounded_slices() {
        let rows = 2 * MATERIALIZE_BATCH_SIZE as u64 + 1;
        let mut script = ScriptedRemapper::new().sized(2, rows);
        for offset in 0..rows {
            script = script.moved(address(2, offset), address(4, offset));
        }
        let (remapper, script) = script.into_remapper();
        let remap = remapper
            .materialize(Some(&RoaringBitmap::from_iter([2u32])))
            .await
            .unwrap();
        assert_eq!(script.batches(), 3);
        assert_eq!(
            remap.get(address(2, rows - 1)),
            Some(Some(address(4, rows - 1)))
        );
    }

    #[test]
    fn materialization_charges_the_rounded_hash_table() {
        // 7,350,000 entries adjust to 8,400,000 and round to 16,777,216
        // buckets of 25 bytes: 419,430,416 bytes, not 32 bytes per entry.
        assert_eq!(hash_map_bytes(7_350_000), 419_430_416);
        assert_eq!(hash_map_bytes(0), 4 * 25 + 16);
        assert!(hash_map_bytes(7_350_000) > DEFAULT_MATERIALIZATION_BUDGET_BYTES);
    }

    #[tokio::test]
    async fn an_unscripted_address_of_a_rewritten_fragment_fails() {
        let (remapper, _) = script();
        let mut addrs = vec![Some(address(1, 9))];
        let error = remapper.translate_in_place(&mut addrs).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }
}
