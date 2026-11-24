// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_select::{RowAddrMask, RowAddrTreeMap};

/// Identity of the deletion state a [`PreFilter`] masks out.
///
/// Two filters with equal fingerprints select exactly the same rows, so a mask
/// resolved from one can answer queries carrying the other. It is a BLAKE3
/// digest of every input the mask is computed from; see
/// [`PreFilter::deletion_fingerprint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaskFingerprint([u8; 32]);

impl MaskFingerprint {
    pub fn new(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// The local ids of one partition that a [`PreFilter`] allows, one bit per id.
///
/// Resolving a filter against a partition walks every row of the partition, so
/// a mask resolved once and kept with the partition lets each query test an id
/// with a single bit read instead.
#[derive(Clone)]
pub struct AllowedMask {
    words: Box<[u64]>,
    len: usize,
    num_allowed: usize,
}

impl AllowedMask {
    /// Builds the mask of a partition with `len` local ids from the ascending
    /// local ids it allows, as [`PreFilter::filter_row_ids`] returns them.
    pub fn from_selected(len: usize, selected: &[u64]) -> Result<Self> {
        let mut words = vec![0u64; len.div_ceil(64)].into_boxed_slice();
        for &id in selected {
            if id as usize >= len {
                return Err(lance_core::Error::invalid_input(format!(
                    "allowed id {id} is out of range for a partition of {len} rows"
                )));
            }
            words[id as usize / 64] |= 1 << (id % 64);
        }
        let num_allowed = words.iter().map(|word| word.count_ones() as usize).sum();
        Ok(Self {
            words,
            len,
            num_allowed,
        })
    }

    /// Number of local ids the mask covers, allowed or not.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn num_allowed(&self) -> usize {
        self.num_allowed
    }

    pub fn is_all_allowed(&self) -> bool {
        self.num_allowed == self.len
    }

    #[inline(always)]
    pub fn contains(&self, id: u32) -> bool {
        self.words[id as usize / 64] & (1 << (id % 64)) != 0
    }

    /// Allowed ids in ascending order.
    pub fn iter_allowed(&self) -> impl Iterator<Item = u32> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, &word)| {
                let base = (word_index * 64) as u32;
                std::iter::successors((word != 0).then_some(word), |rest| {
                    let rest = rest & (rest - 1);
                    (rest != 0).then_some(rest)
                })
                .map(move |rest| base + rest.trailing_zeros())
            })
    }

    /// The allowed ids of rank `0, stride, 2 * stride, ...`, at most `count` of
    /// them, without visiting the allowed ids in between: whole words are
    /// skipped by their population count.
    pub fn allowed_at_stride(&self, stride: usize, count: usize) -> Vec<u32> {
        let stride = stride.max(1);
        let mut picked = Vec::with_capacity(count.min(self.num_allowed));
        // Rank, among the allowed ids, of the next id to pick.
        let mut next_rank = 0usize;
        let mut rank = 0usize;
        for (word_index, &word) in self.words.iter().enumerate() {
            if picked.len() == count {
                break;
            }
            let ones = word.count_ones() as usize;
            while picked.len() < count && next_rank < rank + ones {
                let mut rest = word;
                for _ in 0..next_rank - rank {
                    rest &= rest - 1;
                }
                picked.push((word_index * 64) as u32 + rest.trailing_zeros());
                next_rank += stride;
            }
            rank += ones;
        }
        picked
    }

    /// Bytes the mask holds on the heap.
    pub fn heap_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }
}

impl std::fmt::Debug for AllowedMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllowedMask")
            .field("len", &self.len)
            .field("num_allowed", &self.num_allowed)
            .finish()
    }
}

/// A trait to be implemented by anything supplying a prefilter row addr mask
///
/// This trait is for internal use only and has no stability guarantees.
#[async_trait]
pub trait FilterLoader: Send + 'static {
    /// The mask is shared so that a loader which already holds one behind an
    /// `Arc` (a cached mask, say) hands it out without a copy.
    async fn load(self: Box<Self>) -> Result<Arc<RowAddrMask>>;
}

/// Filter out row ids that we know are not relevant to the query.
///
/// This could be both rows that are deleted or a prefilter
/// that should be applied to the search
///
/// <section class="warning">
/// Internal use only. No API stability guarantees.
/// </section>
#[async_trait]
pub trait PreFilter: Send + Sync {
    /// Waits for the prefilter to be fully loaded
    ///
    /// The prefilter loads in the background while the rest of the index
    /// search is running.  When you are ready to use the prefilter you
    /// must first call this method to ensure it is fully loaded.  This
    /// allows `filter_row_ids` to be a synchronous method.
    async fn wait_for_ready(&self) -> Result<()>;

    /// If the filter is empty.
    fn is_empty(&self) -> bool;

    /// Whether partition-local row coverage is needed to determine if this filter
    /// can be replaced by [`NoFilter`].
    ///
    /// Most filters cannot make this proof and must not make an IVF search
    /// enumerate a partition's row addresses just to call [`Self::is_empty_for`].
    fn needs_partition_row_ids(&self) -> bool {
        false
    }

    /// Whether this filter selects every row in a known partition.
    ///
    /// Callers must first call [`Self::wait_for_ready`]. Implementations that
    /// cannot prove partition-local emptiness fall back to the global answer.
    fn is_empty_for(&self, _rows: &RowAddrTreeMap) -> bool {
        self.is_empty()
    }

    /// Get the row addr mask for this prefilter
    ///
    /// This method must be called after `wait_for_ready`
    fn mask(&self) -> Arc<RowAddrMask>;

    /// Whether the loaded filter proves that no row at all can pass it.
    ///
    /// This is the opposite end of the scale from [`Self::is_empty`], which
    /// says the filter restricts nothing. It is true only once the mask is
    /// known to be an empty allow list, so it must be read after
    /// `wait_for_ready`; before that, and for a block list of any size, it
    /// is false. Index searches use it to skip candidate scoring for a
    /// segment whose own mask selects nothing, so implementations answer
    /// from their stored mask without scanning it.
    fn selects_nothing(&self) -> bool {
        false
    }

    /// Check whether a slice of row ids should be included in a query.
    ///
    /// Returns a vector of indices into the input slice that should be included,
    /// also known as a selection vector.
    ///
    /// This method must be called after `wait_for_ready`
    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64>;

    /// Fingerprint of this filter's mask when the mask is a function of
    /// deletion state alone, so it can be resolved once per partition and
    /// reused by every query carrying the same fingerprint.
    ///
    /// `None` when the filter carries per-query state (a user filter) or cannot
    /// name every input of its mask. Returning `Some` for a mask the
    /// fingerprint does not fully determine would let a query see rows deleted
    /// after the mask was cached.
    fn deletion_fingerprint(&self) -> Option<MaskFingerprint> {
        None
    }

    /// This filter already resolved against the partition being searched,
    /// over the partition's local ids. `None` unless the filter is an
    /// [`AllowedMaskFilter`].
    fn allowed_mask(&self) -> Option<&AllowedMask> {
        None
    }
}

/// A [`PreFilter`] paired with its mask resolved against one partition.
///
/// Every [`PreFilter`] method answers from `inner`; [`PreFilter::allowed_mask`]
/// additionally exposes the resolved mask, which a sub-index can use instead of
/// resolving `inner` against the partition again.
pub struct AllowedMaskFilter {
    inner: Arc<dyn PreFilter>,
    allowed: Arc<AllowedMask>,
}

impl AllowedMaskFilter {
    /// `allowed` must be `inner` resolved against the partition this filter is
    /// handed to, in that partition's local id order.
    pub fn new(inner: Arc<dyn PreFilter>, allowed: Arc<AllowedMask>) -> Self {
        Self { inner, allowed }
    }
}

#[async_trait]
impl PreFilter for AllowedMaskFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        self.inner.wait_for_ready().await
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn needs_partition_row_ids(&self) -> bool {
        self.inner.needs_partition_row_ids()
    }

    fn is_empty_for(&self, rows: &RowAddrTreeMap) -> bool {
        self.inner.is_empty_for(rows)
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        self.inner.mask()
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        self.inner.filter_row_ids(row_ids)
    }

    fn deletion_fingerprint(&self) -> Option<MaskFingerprint> {
        self.inner.deletion_fingerprint()
    }

    fn allowed_mask(&self) -> Option<&AllowedMask> {
        Some(&self.allowed)
    }
}

/// A prefilter that does nothing
pub struct NoFilter;

#[async_trait]
impl PreFilter for NoFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        true
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        Arc::new(RowAddrMask::all_rows())
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        row_ids.enumerate().map(|(i, _)| i as u64).collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::AllowedMask;

    fn mask(len: usize, selected: &[u64]) -> AllowedMask {
        AllowedMask::from_selected(len, selected).unwrap()
    }

    #[test]
    fn test_allowed_mask_membership_and_counts() {
        let selected = [0, 3, 63, 64, 65, 129];
        let allowed = mask(130, &selected);
        assert_eq!(allowed.len(), 130);
        assert_eq!(allowed.num_allowed(), selected.len());
        assert!(!allowed.is_all_allowed());
        for id in 0..130u32 {
            assert_eq!(
                allowed.contains(id),
                selected.contains(&(id as u64)),
                "{id}"
            );
        }
        assert_eq!(
            allowed.iter_allowed().collect::<Vec<_>>(),
            selected.iter().map(|id| *id as u32).collect::<Vec<_>>()
        );

        let all: Vec<u64> = (0..70).collect();
        assert!(mask(70, &all).is_all_allowed());
        assert!(mask(0, &[]).is_all_allowed());
        assert_eq!(mask(70, &[]).iter_allowed().count(), 0);
    }

    #[test]
    fn test_allowed_mask_rejects_out_of_range_id() {
        let error = AllowedMask::from_selected(10, &[3, 10]).unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("id 10"), "{error}");
    }

    /// `allowed_at_stride` must pick exactly what stepping through every allowed
    /// id would, since ACORN seeds its frontier from it.
    #[rstest]
    #[case::stride_one(1, 16)]
    #[case::stride_small(3, 16)]
    #[case::stride_word(64, 16)]
    #[case::stride_large(1000, 16)]
    #[case::count_zero(5, 0)]
    fn test_allowed_at_stride_matches_step_by(#[case] stride: usize, #[case] count: usize) {
        let selected: Vec<u64> = (0..5000u64).filter(|id| (id * 7919) % 13 < 5).collect();
        let allowed = mask(5000, &selected);
        let expected: Vec<u32> = allowed.iter_allowed().step_by(stride).take(count).collect();
        assert_eq!(allowed.allowed_at_stride(stride, count), expected);
    }
}
