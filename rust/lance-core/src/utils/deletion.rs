// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, ops::Range, sync::Arc};

use crate::deepsize::{Context, DeepSizeOf};
use arrow_array::BooleanArray;
use arrow_buffer::BooleanBufferBuilder;
use roaring::RoaringBitmap;

/// Threshold for when a DeletionVector::Set should be promoted to a DeletionVector::Bitmap.
const BITMAP_THRESDHOLD: usize = 5_000;
// TODO: Benchmark to find a better value.

/// Represents a set of deleted row offsets in a single fragment.
#[derive(Debug, Clone, Default)]
pub enum DeletionVector {
    #[default]
    NoDeletions,
    Set(HashSet<u32>),
    Bitmap(RoaringBitmap),
}

impl DeepSizeOf for DeletionVector {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            Self::NoDeletions => 0,
            Self::Set(set) => set.deep_size_of_children(context),
            // Inexact but probably close enough
            Self::Bitmap(bitmap) => bitmap.serialized_size(),
        }
    }
}

impl DeletionVector {
    pub fn len(&self) -> usize {
        match self {
            Self::NoDeletions => 0,
            Self::Set(set) => set.len(),
            Self::Bitmap(bitmap) => bitmap.len() as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, i: u32) -> bool {
        match self {
            Self::NoDeletions => false,
            Self::Set(set) => set.contains(&i),
            Self::Bitmap(bitmap) => bitmap.contains(i),
        }
    }

    pub fn contains_range(&self, mut range: Range<u32>) -> bool {
        match self {
            Self::NoDeletions => range.is_empty(),
            Self::Set(set) => range.all(|i| set.contains(&i)),
            Self::Bitmap(bitmap) => bitmap.contains_range(range),
        }
    }

    fn range_cardinality(&self, range: Range<u32>) -> u64 {
        match self {
            Self::NoDeletions => 0,
            Self::Set(set) => range.fold(0, |acc, i| acc + set.contains(&i) as u64),
            Self::Bitmap(bitmap) => bitmap.range_cardinality(range),
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = u32> + Send + '_> {
        match self {
            Self::NoDeletions => Box::new(std::iter::empty()),
            Self::Set(set) => Box::new(set.iter().copied()),
            Self::Bitmap(bitmap) => Box::new(bitmap.iter()),
        }
    }

    pub fn into_sorted_iter(self) -> Box<dyn Iterator<Item = u32> + Send + 'static> {
        match self {
            Self::NoDeletions => Box::new(std::iter::empty()),
            Self::Set(set) => {
                // If we're using a set we shouldn't have too many values
                // and so this conversion should be affordable.
                let mut values = Vec::from_iter(set);
                values.sort();
                Box::new(values.into_iter())
            }
            // Bitmaps always iterate in sorted order
            Self::Bitmap(bitmap) => Box::new(bitmap.into_iter()),
        }
    }

    /// Create an iterator that iterates over the values in the deletion vector in sorted order.
    pub fn to_sorted_iter<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + Send + 'a> {
        match self {
            Self::NoDeletions => Box::new(std::iter::empty()),
            // We have to make a clone when we're using a set
            // but sets should be relatively small.
            Self::Set(_) => self.clone().into_sorted_iter(),
            Self::Bitmap(bitmap) => Box::new(bitmap.iter()),
        }
    }

    /// Build a mask of the rows in a batch that survive deletion.
    ///
    /// `ranges` are the fragment row offsets the batch covers, in the order the batch
    /// returns them, and must sum to `num_rows`.  Bit `i` of the result is set when the
    /// batch's `i`th row is still live.
    ///
    /// Returns `None` when the batch contains no deleted row, letting the caller pass it
    /// along untouched.
    ///
    /// Unlike [`Self::build_predicate`] this looks deleted rows up by offset instead of
    /// asking about every row in turn, so the cost follows the deletions a batch actually
    /// covers rather than the number of rows it holds.  A [`HashSet`] cannot be queried by
    /// range, so for that representation the two loops are compared by length and the
    /// shorter one is walked.
    pub fn build_keep_mask(&self, ranges: &[Range<u32>], num_rows: u32) -> Option<BooleanArray> {
        debug_assert_eq!(
            ranges.iter().map(|r| r.end - r.start).sum::<u32>(),
            num_rows,
            "ranges must cover exactly num_rows rows"
        );

        let mut mask = KeepMaskBuilder::new(num_rows);
        match self {
            Self::NoDeletions => {}
            Self::Bitmap(bitmap) => {
                let mut base = 0;
                for range in ranges {
                    for deleted in bitmap.range(range.clone()) {
                        mask.delete(base + deleted - range.start);
                    }
                    base += range.end - range.start;
                }
            }
            // Walking the set costs one step per entry for every range, while probing
            // costs one lookup per row.  A set never grows past `BITMAP_THRESDHOLD`, so
            // on a batch-sized read the first is usually the shorter walk.
            Self::Set(set) if set.len().saturating_mul(ranges.len()) <= num_rows as usize => {
                let mut base = 0;
                for range in ranges {
                    for deleted in set.iter().copied().filter(|offset| range.contains(offset)) {
                        mask.delete(base + deleted - range.start);
                    }
                    base += range.end - range.start;
                }
            }
            Self::Set(set) => {
                let mut position = 0;
                for range in ranges {
                    for offset in range.clone() {
                        if set.contains(&offset) {
                            mask.delete(position);
                        }
                        position += 1;
                    }
                }
            }
        }
        mask.finish()
    }

    // Note: deletion vectors are based on 32-bit offsets.  However, this function works
    // even when given 64-bit row addresses.  That is because `id as u32` returns the lower
    // 32 bits (the row offset) and the upper 32 bits are ignored.
    pub fn build_predicate(&self, row_addrs: std::slice::Iter<u64>) -> Option<BooleanArray> {
        match self {
            Self::Bitmap(bitmap) => Some(
                row_addrs
                    .map(|&id| !bitmap.contains(id as u32))
                    .collect::<Vec<_>>(),
            ),
            Self::Set(set) => Some(
                row_addrs
                    .map(|&id| !set.contains(&(id as u32)))
                    .collect::<Vec<_>>(),
            ),
            Self::NoDeletions => None,
        }
        .map(BooleanArray::from)
    }
}

/// Collects deleted positions into a keep mask, allocating only once one shows up.
///
/// Most batches of a scan contain no deleted row at all, and those cost nothing here.
struct KeepMaskBuilder {
    num_rows: u32,
    builder: Option<BooleanBufferBuilder>,
}

impl KeepMaskBuilder {
    fn new(num_rows: u32) -> Self {
        Self {
            num_rows,
            builder: None,
        }
    }

    fn delete(&mut self, position: u32) {
        let num_rows = self.num_rows as usize;
        self.builder
            .get_or_insert_with(|| {
                let mut builder = BooleanBufferBuilder::new(num_rows);
                builder.append_n(num_rows, true);
                builder
            })
            .set_bit(position as usize, false);
    }

    fn finish(self) -> Option<BooleanArray> {
        self.builder
            .map(|mut builder| BooleanArray::new(builder.finish(), None))
    }
}

/// Maps a naive offset into a fragment to the local row offset that is
/// not deleted.
///
/// For example, if the deletion vector is [0, 1, 2], then the mapping
/// would be:
///
/// - 0 -> 3
/// - 1 -> 4
/// - 2 -> 5
///
/// and so on.
///
/// This expects a monotonically increasing sequence of input offsets. State
/// is re-used between calls to `map_offset` to make the mapping more efficient.
pub struct OffsetMapper {
    dv: Arc<DeletionVector>,
    left: u32,
    last_diff: u32,
}

impl OffsetMapper {
    pub fn new(dv: Arc<DeletionVector>) -> Self {
        Self {
            dv,
            left: 0,
            last_diff: 0,
        }
    }

    pub fn map_offset(&mut self, offset: u32) -> u32 {
        // The best initial guess is the offset + last diff. That's the right
        // answer if there are no deletions in the range between the last
        // offset and the current one.
        let mut mid = offset + self.last_diff;
        let mut right = offset + self.dv.len() as u32;
        loop {
            let deleted_in_range = self.dv.range_cardinality(0..(mid + 1)) as u32;
            match mid.cmp(&(offset + deleted_in_range)) {
                std::cmp::Ordering::Equal if !self.dv.contains(mid) => {
                    self.last_diff = mid - offset;
                    return mid;
                }
                std::cmp::Ordering::Less => {
                    assert_ne!(self.left, mid + 1);
                    self.left = mid + 1;
                    mid = self.left + (right - self.left) / 2;
                }
                // Binary search left when the guess overshoots. This can happen when:
                // - Greater: last_diff was calibrated for a denser deletion region
                // - Equal with deleted mid: the guess lands exactly on a deleted row
                std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => {
                    right = mid;
                    mid = self.left + (right - self.left) / 2;
                }
            }
        }
    }
}

impl From<&DeletionVector> for RoaringBitmap {
    fn from(value: &DeletionVector) -> Self {
        match value {
            DeletionVector::Bitmap(bitmap) => bitmap.clone(),
            DeletionVector::Set(set) => Self::from_iter(set.iter()),
            DeletionVector::NoDeletions => Self::new(),
        }
    }
}

impl PartialEq for DeletionVector {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NoDeletions, Self::NoDeletions) => true,
            (Self::Set(set1), Self::Set(set2)) => set1 == set2,
            (Self::Bitmap(bitmap1), Self::Bitmap(bitmap2)) => bitmap1 == bitmap2,
            (Self::Set(set), Self::Bitmap(bitmap)) | (Self::Bitmap(bitmap), Self::Set(set)) => {
                let set = set.iter().copied().collect::<RoaringBitmap>();
                set == *bitmap
            }
            _ => false,
        }
    }
}

impl Extend<u32> for DeletionVector {
    fn extend<T: IntoIterator<Item = u32>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        // The mem::replace allows changing the variant of Self when we only
        // have &mut Self.
        *self = match (std::mem::take(self), iter.size_hint()) {
            (Self::NoDeletions, (_, Some(0))) => Self::NoDeletions,
            (Self::NoDeletions, (lower, _)) if lower >= BITMAP_THRESDHOLD => {
                let bitmap = iter.collect::<RoaringBitmap>();
                Self::Bitmap(bitmap)
            }
            (Self::NoDeletions, (_, Some(upper))) if upper < BITMAP_THRESDHOLD => {
                let set = iter.collect::<HashSet<_>>();
                Self::Set(set)
            }
            (Self::NoDeletions, _) => {
                // We don't know the size, so just try as a set and move to bitmap
                // if it ends up being big.
                let set = iter.collect::<HashSet<_>>();
                if set.len() > BITMAP_THRESDHOLD {
                    let bitmap = set.into_iter().collect::<RoaringBitmap>();
                    Self::Bitmap(bitmap)
                } else {
                    Self::Set(set)
                }
            }
            (Self::Set(mut set), _) => {
                set.extend(iter);
                if set.len() > BITMAP_THRESDHOLD {
                    let bitmap = set.drain().collect::<RoaringBitmap>();
                    Self::Bitmap(bitmap)
                } else {
                    Self::Set(set)
                }
            }
            (Self::Bitmap(mut bitmap), _) => {
                bitmap.extend(iter);
                Self::Bitmap(bitmap)
            }
        };
    }
}

// TODO: impl methods for DeletionVector
/// impl DeletionVector {
///     pub fn get(i: u32) -> bool { ... }
/// }
/// impl BitAnd for DeletionVector { ... }
impl IntoIterator for DeletionVector {
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + Send>;
    type Item = u32;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::NoDeletions => Box::new(std::iter::empty()),
            Self::Set(set) => {
                // In many cases, it's much better if this is sorted. It's
                // guaranteed to be small, so the cost is low.
                let mut sorted = set.into_iter().collect::<Vec<_>>();
                sorted.sort();
                Box::new(sorted.into_iter())
            }
            Self::Bitmap(bitmap) => Box::new(bitmap.into_iter()),
        }
    }
}

impl FromIterator<u32> for DeletionVector {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let mut deletion_vector = Self::default();
        deletion_vector.extend(iter);
        deletion_vector
    }
}

impl From<RoaringBitmap> for DeletionVector {
    fn from(bitmap: RoaringBitmap) -> Self {
        if bitmap.is_empty() {
            Self::NoDeletions
        } else {
            Self::Bitmap(bitmap)
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod test {
    use super::*;
    use crate::deepsize::DeepSizeOf;
    use rstest::rstest;

    fn set_dv(vals: impl IntoIterator<Item = u32>) -> DeletionVector {
        DeletionVector::Set(HashSet::from_iter(vals))
    }
    fn bitmap_dv(vals: impl IntoIterator<Item = u32>) -> DeletionVector {
        DeletionVector::Bitmap(RoaringBitmap::from_iter(vals))
    }

    #[test]
    fn test_set_bitmap_equality() {
        assert_eq!(set_dv(0..100), bitmap_dv(0..100));
    }

    #[test]
    fn test_threshold_promotes_to_bitmap() {
        let dv = DeletionVector::from_iter(0..(BITMAP_THRESDHOLD as u32));
        assert!(matches!(dv, DeletionVector::Bitmap(_)));
    }

    #[rstest]
    #[case::middle_deletions(&[3, 5], &[0, 1, 2, 4, 6, 7, 8])]
    #[case::start_deletions(&[0, 1, 2], &[3, 4, 5, 6, 7, 8, 9])]
    fn test_map_offsets(#[case] deleted: &[u32], #[case] expected: &[u32]) {
        let dv = DeletionVector::from_iter(deleted.iter().copied());
        let mut mapper = OffsetMapper::new(Arc::new(dv));
        let output: Vec<_> = (0..expected.len() as u32)
            .map(|o| mapper.map_offset(o))
            .collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_deep_size_of() {
        assert_eq!(
            DeletionVector::NoDeletions.deep_size_of(),
            std::mem::size_of::<DeletionVector>()
        );
        assert!(set_dv([1, 2, 3]).deep_size_of() > std::mem::size_of::<DeletionVector>());
        assert!(bitmap_dv([1, 2, 3]).deep_size_of() > std::mem::size_of::<DeletionVector>());
    }

    #[rstest]
    #[case::no_deletions(DeletionVector::NoDeletions, 0, true)]
    #[case::set(set_dv([1, 2, 3]), 3, false)]
    #[case::bitmap(bitmap_dv([1, 2, 3, 4, 5]), 5, false)]
    fn test_len_is_empty(#[case] dv: DeletionVector, #[case] len: usize, #[case] empty: bool) {
        assert_eq!(dv.len(), len);
        assert_eq!(dv.is_empty(), empty);
    }

    #[rstest]
    #[case::no_deletions(DeletionVector::NoDeletions, 1, false)]
    #[case::set_contains(set_dv([1, 2, 3]), 1, true)]
    #[case::set_missing(set_dv([1, 2, 3]), 0, false)]
    #[case::bitmap_contains(bitmap_dv([10, 20, 30]), 10, true)]
    #[case::bitmap_missing(bitmap_dv([10, 20, 30]), 5, false)]
    fn test_contains(#[case] dv: DeletionVector, #[case] val: u32, #[case] expected: bool) {
        assert_eq!(dv.contains(val), expected);
    }

    #[rstest]
    #[case::no_del_empty_range(DeletionVector::NoDeletions, 0..0, true)]
    #[case::no_del_non_empty(DeletionVector::NoDeletions, 0..1, false)]
    #[case::set_full_range(set_dv([1, 2, 3]), 1..4, true)]
    #[case::set_partial(set_dv([1, 2, 3]), 0..2, false)]
    #[case::bitmap_full(bitmap_dv([10, 11, 12]), 10..13, true)]
    #[case::bitmap_partial(bitmap_dv([10, 11, 12]), 9..11, false)]
    fn test_contains_range(
        #[case] dv: DeletionVector,
        #[case] range: std::ops::Range<u32>,
        #[case] expected: bool,
    ) {
        assert_eq!(dv.contains_range(range), expected);
    }

    #[test]
    fn test_range_cardinality() {
        assert_eq!(DeletionVector::NoDeletions.range_cardinality(0..100), 0);
        let bm = bitmap_dv([5, 10, 15]);
        assert_eq!(bm.range_cardinality(0..20), 3);
        assert_eq!(bm.range_cardinality(6..14), 1);
    }

    #[rstest]
    #[case::no_deletions(DeletionVector::NoDeletions, vec![])]
    #[case::set(set_dv([3, 1, 2]), vec![1, 2, 3])]
    #[case::bitmap(bitmap_dv([30, 10, 20]), vec![10, 20, 30])]
    fn test_iterators(#[case] dv: DeletionVector, #[case] expected: Vec<u32>) {
        // Test iter()
        let mut items: Vec<_> = dv.iter().collect();
        items.sort();
        assert_eq!(items, expected);

        // Test to_sorted_iter()
        assert_eq!(dv.to_sorted_iter().collect::<Vec<_>>(), expected);

        // Test into_sorted_iter() and into_iter() (both consume, so clone first)
        assert_eq!(dv.clone().into_sorted_iter().collect::<Vec<_>>(), expected);
        assert_eq!(dv.into_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_build_predicate() {
        let addrs = [0u64, 1, 2, 3, 4];
        assert!(
            DeletionVector::NoDeletions
                .build_predicate(addrs.iter())
                .is_none()
        );

        let pred = set_dv([1, 3]).build_predicate(addrs.iter()).unwrap();
        assert_eq!(
            pred.iter().map(|v| v.unwrap()).collect::<Vec<_>>(),
            [true, false, true, false, true]
        );

        let pred = bitmap_dv([0, 2, 4]).build_predicate(addrs.iter()).unwrap();
        assert_eq!(
            pred.iter().map(|v| v.unwrap()).collect::<Vec<_>>(),
            [false, true, false, true, false]
        );
    }

    /// A missing mask means every row survived, so spell that out as a full mask.
    fn keep_mask_values(dv: &DeletionVector, ranges: &[Range<u32>], num_rows: u32) -> Vec<bool> {
        match dv.build_keep_mask(ranges, num_rows) {
            Some(mask) => mask.iter().map(|v| v.unwrap()).collect(),
            None => vec![true; num_rows as usize],
        }
    }

    #[rstest]
    #[case::set(set_dv([1, 3]))]
    #[case::bitmap(bitmap_dv([1, 3]))]
    fn test_build_keep_mask(#[case] dv: DeletionVector) {
        assert_eq!(
            keep_mask_values(&dv, &[0..5], 5),
            [true, false, true, false, true]
        );
    }

    #[rstest]
    #[case::set(set_dv([1, 6, 9]))]
    #[case::bitmap(bitmap_dv([1, 6, 9]))]
    fn test_build_keep_mask_maps_offsets_to_batch_positions(#[case] dv: DeletionVector) {
        // Offsets 0..3 land at positions 0..3 and offsets 6..10 at positions 3..7, so the
        // deleted offsets 1, 6 and 9 must show up at positions 1, 3 and 6.
        assert_eq!(
            keep_mask_values(&dv, &[0..3, 6..10], 7),
            [true, false, true, false, true, true, false]
        );
    }

    #[rstest]
    #[case::no_deletions(DeletionVector::NoDeletions)]
    #[case::set(set_dv([100, 200]))]
    #[case::bitmap(bitmap_dv([100, 200]))]
    fn test_build_keep_mask_skips_batches_without_deletions(#[case] dv: DeletionVector) {
        assert!(dv.build_keep_mask(&[0..10], 10).is_none());
    }

    /// The row addresses `build_predicate` would be handed for a batch covering `ranges`.
    fn batch_row_addrs(ranges: &[Range<u32>]) -> Vec<u64> {
        ranges
            .iter()
            .flat_map(|range| range.clone())
            .map(u64::from)
            .collect()
    }

    #[rstest]
    // Few enough deleted rows that walking the set is the shorter loop.
    #[case::set_walked(set_dv([2, 5, 11]), vec![0..8, 10..14], 12)]
    // More deleted rows than the batch holds, so the per-row probe is used instead.
    #[case::set_probed(set_dv([0, 1, 2, 3, 4, 5, 6, 7]), vec![2..5], 3)]
    #[case::bitmap(bitmap_dv([2, 5, 11]), vec![0..8, 10..14], 12)]
    fn test_build_keep_mask_matches_build_predicate(
        #[case] dv: DeletionVector,
        #[case] ranges: Vec<Range<u32>>,
        #[case] num_rows: u32,
    ) {
        let row_addrs = batch_row_addrs(&ranges);
        let expected = dv
            .build_predicate(row_addrs.iter())
            .map(|mask| mask.iter().map(|v| v.unwrap()).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![true; num_rows as usize]);
        assert_eq!(keep_mask_values(&dv, &ranges, num_rows), expected);
    }

    /// Bitmap lookups always go through `range()`, including short runs, runs that
    /// start mid-fragment, and a batch that mixes several of those in one call.
    #[rstest]
    #[case::short_run(vec![0..3])]
    #[case::long_run(vec![0..64])]
    #[case::offset_run(vec![32..80])]
    #[case::mixed_runs(vec![0..3, 10..26, 100..104])]
    fn test_build_keep_mask_range_lookup(#[case] ranges: Vec<Range<u32>>) {
        let num_rows = ranges.iter().map(|range| range.end - range.start).sum();
        // Spread across the fragment and onto run edges, so every case drops a row.
        let dv = bitmap_dv([0, 3, 11, 25, 40, 41, 42, 79, 100, 103]);

        let row_addrs = batch_row_addrs(&ranges);
        let expected = dv
            .build_predicate(row_addrs.iter())
            .map(|mask| mask.iter().map(|v| v.unwrap()).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![true; num_rows as usize]);
        assert!(
            expected.contains(&false),
            "case would pass without resolving any deletion"
        );
        assert_eq!(keep_mask_values(&dv, &ranges, num_rows), expected);
    }

    #[rstest]
    #[case::no_deletions(DeletionVector::NoDeletions, 0)]
    #[case::set(set_dv([1, 2, 3]), 3)]
    #[case::bitmap(bitmap_dv([10, 20]), 2)]
    fn test_to_roaring(#[case] dv: DeletionVector, #[case] len: u64) {
        let bitmap: RoaringBitmap = (&dv).into();
        assert_eq!(bitmap.len(), len);
    }

    #[test]
    fn test_partial_eq() {
        assert_eq!(DeletionVector::NoDeletions, DeletionVector::NoDeletions);
        assert_eq!(set_dv([1, 2, 3]), set_dv([1, 2, 3]));
        assert_eq!(bitmap_dv([1, 2, 3]), bitmap_dv([1, 2, 3]));
        assert_eq!(set_dv([5, 6, 7]), bitmap_dv([5, 6, 7])); // cross-type
        assert_eq!(bitmap_dv([5, 6, 7]), set_dv([5, 6, 7])); // reverse
        assert_ne!(DeletionVector::NoDeletions, set_dv([1]));
        assert_ne!(DeletionVector::NoDeletions, bitmap_dv([1]));
    }

    #[test]
    fn test_extend() {
        // Empty iter -> stays NoDeletions
        let mut dv = DeletionVector::NoDeletions;
        dv.extend(std::iter::empty::<u32>());
        assert!(matches!(dv, DeletionVector::NoDeletions));

        // Unknown size small -> Set
        let mut dv = DeletionVector::NoDeletions;
        dv.extend(std::iter::from_fn({
            let mut i = 0u32;
            move || {
                i += 1;
                (i <= 10).then_some(i - 1)
            }
        }));
        assert!(matches!(dv, DeletionVector::Set(_)));

        // Unknown size large -> Bitmap
        let mut dv = DeletionVector::NoDeletions;
        dv.extend((0u32..10_000).filter(|_| true));
        assert!(matches!(dv, DeletionVector::Bitmap(_)));

        // Set stays Set when small
        let mut dv = set_dv([1, 2, 3]);
        dv.extend([4, 5, 6]);
        assert!(matches!(dv, DeletionVector::Set(_)) && dv.len() == 6);

        // Set promotes to Bitmap when large
        let mut dv = set_dv([1, 2, 3]);
        dv.extend(100..(BITMAP_THRESDHOLD as u32 + 100));
        assert!(matches!(dv, DeletionVector::Bitmap(_)));

        // Bitmap stays Bitmap
        let mut dv = bitmap_dv([1, 2, 3]);
        dv.extend([4, 5, 6]);
        assert!(matches!(dv, DeletionVector::Bitmap(_)) && dv.len() == 6);
    }

    #[test]
    fn test_from_roaring() {
        let dv: DeletionVector = RoaringBitmap::new().into();
        assert!(matches!(dv, DeletionVector::NoDeletions));

        let dv: DeletionVector = RoaringBitmap::from_iter([1, 2, 3]).into();
        assert!(matches!(dv, DeletionVector::Bitmap(_)) && dv.len() == 3);
    }

    #[test]
    fn test_map_offset_dense_then_sparse() {
        // First half densely deleted (80% deleted), second half sparse (20% deleted)
        // This creates varying deletion density that might trip up the algorithm
        let mut deleted = Vec::new();
        // Dense region: delete 4 out of every 5 rows (keep every 5th)
        for i in 0..500u32 {
            if i % 5 != 0 {
                deleted.push(i);
            }
        }
        // Sparse region: delete 1 out of every 5 rows
        for i in 500..1000u32 {
            if i % 5 == 0 {
                deleted.push(i);
            }
        }
        let dv = DeletionVector::Bitmap(RoaringBitmap::from_iter(deleted));
        let mut mapper = OffsetMapper::new(Arc::new(dv));

        // In dense region: offset 0 -> row 0 (kept), offset 1 -> row 5 (kept), etc.
        assert_eq!(mapper.map_offset(0), 0);
        assert_eq!(mapper.map_offset(1), 5);
        assert_eq!(mapper.map_offset(99), 495);

        // Transition to sparse region
        // At row 500, we've had 400 deletions in dense region, plus row 500 is deleted
        // offset 100 should get row 501
        assert_eq!(mapper.map_offset(100), 501);
    }
}
