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

use std::any::Any;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

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

    fn as_any(&self) -> &dyn Any;
}

/// Translates the row addresses one index segment stores into the addresses
/// the rows have in the snapshot being read.
///
/// Cheap to clone. Fragments the remapper does not need to translate keep
/// their addresses; an address whose row was deleted, or that the segment no
/// longer answers for, translates to `None`.
#[derive(Clone, Debug)]
pub struct Remapper(Arc<dyn RemapperBacking>);

impl Remapper {
    #[doc(hidden)]
    pub fn from_backing(backing: Arc<dyn RemapperBacking>) -> Self {
        Self(backing)
    }

    #[doc(hidden)]
    pub fn backing(&self) -> &dyn RemapperBacking {
        self.0.as_ref()
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
        self.translate_in_place(&mut pending).await?;
        for addr in pending.into_iter().flatten() {
            translated.insert(addr);
        }
        Ok(translated)
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

    fn as_any(&self) -> &dyn Any {
        self
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
        let (remapper, _) = script();
        let mut addrs = RowAddrTreeMap::new();
        addrs.insert(address(1, 0));
        addrs.insert(address(1, 1));
        addrs.insert_fragment(2);
        let translated = remapper.translate_set(&addrs).await.unwrap();
        let mut expected = RowAddrTreeMap::new();
        expected.insert(address(3, 1));
        expected.insert_fragment(2);
        assert_eq!(translated, expected);

        let mut whole_rewritten = RowAddrTreeMap::new();
        whole_rewritten.insert_fragment(1);
        let error = remapper.translate_set(&whole_rewritten).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("fragment 1"), "{error}");
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

    #[tokio::test]
    async fn an_unscripted_address_of_a_rewritten_fragment_fails() {
        let (remapper, _) = script();
        let mut addrs = vec![Some(address(1, 9))];
        let error = remapper.translate_in_place(&mut addrs).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }
}
