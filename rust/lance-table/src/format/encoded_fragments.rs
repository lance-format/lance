// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Encoded `DataFragment` messages carried alongside a [`Manifest`].
//!
//! Writing a manifest encodes every fragment. On a manifest with millions of
//! fragments that dominates commit time, even when the transaction only touched
//! a few thousand of them. [`EncodedFragmentCache`] keeps the encoded
//! `pb::DataFragment` bytes of a manifest's fragments, so a manifest derived
//! from it only encodes the fragments its transaction added or changed.
//!
//! The cache describes one specific fragment list: it holds a [`Weak`] pointer
//! to the manifest's `Arc<Vec<Fragment>>` and is ignored for any other list.
//! Replacing `Manifest::fragments`, or mutating it through `Arc::make_mut`
//! (which moves the list to a new allocation while a `Weak` exists), therefore
//! invalidates the cache, and the writer falls back to encoding every fragment.
//!
//! [`Manifest`]: super::Manifest

use std::sync::{Arc, Weak};

use bytes::Bytes;
use lance_core::deepsize::{Context, DeepSizeOf};

use super::Fragment;

/// Encoded `pb::DataFragment` bytes for the fragments of one manifest, by
/// position. `None` marks a fragment that must be encoded at write time.
#[derive(Clone, Default)]
pub struct EncodedFragmentCache {
    inner: Option<EncodedFragments>,
}

#[derive(Clone)]
struct EncodedFragments {
    /// The fragment list `encoded` describes, by identity.
    fragments: Weak<Vec<Fragment>>,
    encoded: Arc<Vec<Option<Bytes>>>,
    /// Size of the buffer the entries of `encoded` slice, which they keep
    /// alive whole.
    retained_bytes: usize,
}

impl EncodedFragmentCache {
    /// A cache for `fragments` holding `encoded[i]` for `fragments[i]`, where
    /// the entries are slices of one buffer of `retained_bytes` bytes.
    ///
    /// Returns an empty cache when the lengths differ or no entry is present.
    pub fn new(
        fragments: &Arc<Vec<Fragment>>,
        encoded: Vec<Option<Bytes>>,
        retained_bytes: usize,
    ) -> Self {
        if encoded.len() != fragments.len() || encoded.iter().all(Option::is_none) {
            return Self::default();
        }
        Self {
            inner: Some(EncodedFragments {
                fragments: Arc::downgrade(fragments),
                encoded: Arc::new(encoded),
                retained_bytes,
            }),
        }
    }

    /// Size of the buffer the cached entries keep alive.
    pub fn retained_bytes(&self) -> usize {
        self.inner.as_ref().map_or(0, |inner| inner.retained_bytes)
    }

    /// The encoded fragments, if this cache describes exactly `fragments`.
    pub fn get(&self, fragments: &Arc<Vec<Fragment>>) -> Option<&[Option<Bytes>]> {
        let inner = self.inner.as_ref()?;
        // While the `Weak` exists the allocation it points to cannot be reused,
        // so pointer equality means "the same list", never a recycled address.
        let same_list = inner.fragments.strong_count() > 0
            && std::ptr::eq(inner.fragments.as_ptr(), Arc::as_ptr(fragments));
        (same_list && inner.encoded.len() == fragments.len()).then(|| inner.encoded.as_slice())
    }
}

/// Encoded bytes for `next[i]`, taken from `previous_encoded` for every
/// fragment whose id is not in `changed_ids`.
///
/// `next` must be sorted by fragment id; `previous` is the list
/// `previous_encoded` describes and is expected to be sorted by id too (an
/// unsorted list only causes misses). `changed_ids` must contain every fragment
/// id whose content in `next` differs from `previous`; it need not be sorted.
pub fn carry_encoded_fragments(
    previous: &[Fragment],
    previous_encoded: &[Option<Bytes>],
    next: &[Fragment],
    changed_ids: &[u64],
) -> Vec<Option<Bytes>> {
    debug_assert_eq!(previous.len(), previous_encoded.len());
    let mut changed_ids = changed_ids.to_vec();
    changed_ids.sort_unstable();
    changed_ids.dedup();

    let mut changed = changed_ids.iter().peekable();
    let mut prev = 0;
    let mut encoded = Vec::with_capacity(next.len());
    for fragment in next {
        while changed.next_if(|&&id| id < fragment.id).is_some() {}
        if changed.peek() == Some(&&fragment.id) {
            encoded.push(None);
            continue;
        }
        while prev < previous.len() && previous[prev].id < fragment.id {
            prev += 1;
        }
        if prev < previous.len() && previous[prev].id == fragment.id {
            debug_assert_eq!(
                &previous[prev], fragment,
                "fragment {} reused encoded bytes but its content changed; \
                 the transaction did not report it as changed",
                fragment.id
            );
            encoded.push(previous_encoded[prev].clone());
            // Step past the match so duplicate ids pair up in order.
            prev += 1;
        } else {
            encoded.push(None);
        }
    }
    encoded
}

impl std::fmt::Debug for EncodedFragmentCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cached = self
            .inner
            .as_ref()
            .map_or(0, |inner| inner.encoded.iter().flatten().count());
        f.debug_struct("EncodedFragmentCache")
            .field("cached", &cached)
            .finish()
    }
}

/// The cache is derived state: two manifests with the same content are equal
/// whether or not either carries encoded fragments.
impl PartialEq for EncodedFragmentCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl DeepSizeOf for EncodedFragmentCache {
    /// The entry list plus the whole buffer the entries keep alive, not just
    /// the bytes they cover.
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.inner.as_ref().map_or(0, |inner| {
            inner.encoded.capacity() * std::mem::size_of::<Option<Bytes>>() + inner.retained_bytes
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fragments(ids: &[u64]) -> Vec<Fragment> {
        ids.iter().map(|&id| Fragment::new(id)).collect()
    }

    fn bytes_for(ids: &[u64]) -> Vec<Option<Bytes>> {
        ids.iter()
            .map(|id| Some(Bytes::from(id.to_le_bytes().to_vec())))
            .collect()
    }

    #[test]
    fn test_cache_is_tied_to_one_fragment_list() {
        let list = Arc::new(fragments(&[1, 2]));
        let cache = EncodedFragmentCache::new(&list, bytes_for(&[1, 2]), 16);
        assert!(cache.get(&list).is_some());

        // A clone of the Arc is the same list.
        let alias = Arc::clone(&list);
        assert!(cache.get(&alias).is_some());
        drop(alias);

        // An equal but distinct list is not.
        let other = Arc::new(fragments(&[1, 2]));
        assert!(cache.get(&other).is_none());

        // make_mut on a uniquely owned list moves it while the cache's Weak
        // exists, so the cache no longer applies.
        let mut list = list;
        Arc::make_mut(&mut list)[0].physical_rows = Some(10);
        assert!(cache.get(&list).is_none());
    }

    #[test]
    fn test_cache_rejects_mismatched_lengths_and_empty_entries() {
        let list = Arc::new(fragments(&[1, 2]));
        assert!(
            EncodedFragmentCache::new(&list, bytes_for(&[1]), 8)
                .get(&list)
                .is_none()
        );
        assert!(
            EncodedFragmentCache::new(&list, vec![None, None], 0)
                .get(&list)
                .is_none()
        );
    }

    #[test]
    fn test_cache_size_counts_the_retained_buffer() {
        let list = Arc::new(fragments(&[1, 2]));
        let buffer = Bytes::from(vec![0u8; 1000]);
        let encoded = vec![Some(buffer.slice(0..10)), Some(buffer.slice(10..20))];
        let cache = EncodedFragmentCache::new(&list, encoded, buffer.len());
        assert_eq!(cache.retained_bytes(), 1000);
        assert_eq!(
            cache.deep_size_of_children(&mut Context::new()),
            2 * std::mem::size_of::<Option<Bytes>>() + 1000
        );
        assert_eq!(
            EncodedFragmentCache::default().deep_size_of_children(&mut Context::new()),
            0
        );
    }

    #[test]
    fn test_carry_skips_changed_new_and_removed_fragments() {
        let previous = fragments(&[1, 2, 3, 5]);
        let previous_encoded = bytes_for(&[1, 2, 3, 5]);
        // 2 changed, 3 removed, 4 and 6 new.
        let mut next = fragments(&[1, 2, 4, 5, 6]);
        next[1].physical_rows = Some(7);

        let carried = carry_encoded_fragments(&previous, &previous_encoded, &next, &[6, 2, 4]);
        assert_eq!(
            carried,
            vec![
                previous_encoded[0].clone(),
                None,
                None,
                previous_encoded[3].clone(),
                None,
            ]
        );
    }

    #[test]
    fn test_carry_keeps_missing_previous_entries_missing() {
        let previous = fragments(&[1, 2]);
        let previous_encoded = vec![None, Some(Bytes::from_static(b"2"))];
        let carried = carry_encoded_fragments(&previous, &previous_encoded, &previous, &[]);
        assert_eq!(carried, previous_encoded);
    }
}
