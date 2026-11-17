// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::utils::address::RowAddress;

use super::{RowAddrMask, RowAddrSelection, RowAddrTreeMap};

const PREPARED_MAX_BYTES: usize = 1024 * 1024;
const PREPARED_MIN_FRAGMENTS: usize = 32;

#[derive(Clone, Copy, Debug)]
enum IntervalSelection {
    Missing,
    Full,
    Range { first: u32, last: u32 },
}

fn proof_fits_budget(
    slots: usize,
    row_visits: usize,
    serialized_sizes: impl IntoIterator<Item = usize>,
) -> bool {
    if slots > row_visits {
        return false;
    }
    serialized_sizes
        .into_iter()
        .try_fold(slots, |words, bytes| {
            words
                .checked_add(bytes.div_ceil(8))
                .filter(|&total| total <= row_visits)
        })
        .is_some()
}

/// An immutable query-local membership view of an existing row mask.
///
/// This owns the original mask through an [`Arc`], so its private interval
/// certificates cannot become stale if another owner uses `Arc::make_mut`.
/// Preparation is optional: small/sparse masks, insufficient query work, holes
/// or allocation failure retain the original mask path. No persistent format,
/// NULL algebra, row-ID domain or `Partial` selection is changed.
///
/// ```
/// use std::sync::Arc;
/// use lance_core::utils::address::RowAddress;
/// use lance_select::{PreparedRowAddrMask, RowAddrMask};
///
/// let mask = RowAddrMask::from_allowed((0..32).flat_map(|key| {
///     (5..10).map(move |offset| u64::from(RowAddress::new_from_parts(key, offset)))
/// }).collect());
/// let prepared = PreparedRowAddrMask::try_new(Arc::new(mask), [4096]).unwrap();
/// let row = u64::from(RowAddress::new_from_parts(3, 7));
/// assert_eq!(prepared.selected_indices([row, row].iter()), [0, 1]);
/// ```
#[derive(Debug)]
pub struct PreparedRowAddrMask {
    mask: Arc<RowAddrMask>,
    /// Fragment key of `slots[0]`. Compaction assigns new fragments keys above
    /// every existing key, so slots cover only the active key span.
    first_key: u32,
    slots: Vec<IntervalSelection>,
}

/// Slot of `key` in a table whose first slot is `first_key`.
///
/// Keys below the origin wrap past every valid slot (the span cannot exceed
/// `u32::MAX`), so bounds-checked lookups treat them as missing.
#[inline]
fn slot_index(first_key: u32, key: u32) -> usize {
    key.wrapping_sub(first_key) as usize
}

impl PreparedRowAddrMask {
    fn preparation_map(mask: &RowAddrMask) -> Option<(&RowAddrTreeMap, u32, usize)> {
        let RowAddrMask::AllowList(map) = mask else {
            return None;
        };
        let fragments = map.inner.len();
        if fragments < PREPARED_MIN_FRAGMENTS {
            return None;
        }
        let (&first_key, _) = map.inner.first_key_value()?;
        let (&last_key, _) = map.inner.last_key_value()?;
        let slots = usize::try_from(u64::from(last_key - first_key) + 1).ok()?;
        if slots > fragments.checked_mul(2)? {
            return None;
        }
        let bytes = slots.checked_mul(std::mem::size_of::<IntervalSelection>())?;
        if bytes > PREPARED_MAX_BYTES {
            return None;
        }

        // Reject all-Full/empty masks before consuming the caller's lazy counts,
        // but do not enumerate serialized sizes until the usable budget is known.
        let has_partial = map.inner.values().any(
            |selection| matches!(selection, RowAddrSelection::Partial(bits) if !bits.is_empty()),
        );
        has_partial.then_some((map, first_key, slots))
    }

    /// Prepare exact interval certificates once for a shared immutable query mask.
    ///
    /// All nonempty partial sets must be exact contiguous ranges. A holed set
    /// returns `None` for the whole view, preserving the original mixed-mask hot
    /// loop. Slot payload is bounded by the existing 1 MiB dense-lookup limit;
    /// allocation failure also returns `None`. The source mask is never mutated.
    /// Counts of partitions too short for the original selector are excluded
    /// from the setup budget. Unsupported shapes do not consume the iterator.
    pub fn try_new(
        mask: Arc<RowAddrMask>,
        partition_row_counts: impl IntoIterator<Item = usize>,
    ) -> Option<Self> {
        Self::prepare(mask, partition_row_counts)?()
    }

    /// Admit optional preparation before dispatching its CPU-only proof.
    ///
    /// Structural checks and the actual partition-work budget run now, once.
    /// The returned closure owns the immutable mask and performs the allocation
    /// and exact range proofs. Unsupported shapes or insufficient work return
    /// `None` without a CPU dispatch. An admitted proof can still reject holes.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_core::utils::address::RowAddress;
    /// use lance_select::{PreparedRowAddrMask, RowAddrMask};
    /// let mask = Arc::new(RowAddrMask::from_allowed((0..32).flat_map(|key| {
    ///     (0..32).map(move |offset| u64::from(RowAddress::new_from_parts(key, offset)))
    /// }).collect()));
    /// let proof = PreparedRowAddrMask::prepare(mask, [4096]).unwrap();
    /// let prepared = proof().unwrap();
    /// assert!(prepared.selected(RowAddress::new_from_parts(3, 7).into()));
    /// ```
    pub fn prepare(
        mask: Arc<RowAddrMask>,
        partition_row_counts: impl IntoIterator<Item = usize>,
    ) -> Option<impl FnOnce() -> Option<Self> + Send + 'static> {
        // Structural eligibility precedes consuming even a lazy partition iterator.
        let (map, first_key, slot_count) = Self::preparation_map(&mask)?;
        // Structural admission proves slots >= F >= 32. For this exact
        // immutable shape, lower >= slots is therefore precisely its remaining
        // per-partition admission condition, without repeated BTree max lookups.
        let expected_row_visits =
            partition_row_counts
                .into_iter()
                .try_fold(0_usize, |rows, count| {
                    if count >= slot_count {
                        rows.checked_add(count)
                    } else {
                        Some(rows)
                    }
                })?;
        // Charge metadata lazily and stop as soon as setup cannot be amortized.
        // A small probe must not visit every bitmap just to reject preparation.
        // This is a storage-word admission budget, not equal instruction costs.
        let serialized_sizes = map.inner.values().filter_map(|selection| match selection {
            RowAddrSelection::Partial(bits) => Some(bits.serialized_size()),
            RowAddrSelection::Full => None,
        });
        if !proof_fits_budget(slot_count, expected_row_visits, serialized_sizes) {
            return None;
        }
        Some(move || Self::build_admitted(mask, first_key, slot_count))
    }

    fn build_admitted(mask: Arc<RowAddrMask>, first_key: u32, slot_count: usize) -> Option<Self> {
        let RowAddrMask::AllowList(map) = mask.as_ref() else {
            return None;
        };
        let mut slots = Vec::new();
        slots.try_reserve_exact(slot_count).ok()?;
        slots.resize(slot_count, IntervalSelection::Missing);
        for (&key, selection) in &map.inner {
            let certificate = match selection {
                RowAddrSelection::Full => IntervalSelection::Full,
                RowAddrSelection::Partial(bits) => {
                    let Some(first) = bits.min() else {
                        continue;
                    };
                    // len distinct values all inside this len-wide interval prove
                    // exact equality. Avoid bitmap.max(), which may scan a long
                    // zero suffix in each 65536-bit container.
                    let last = u64::from(first).checked_add(bits.len().checked_sub(1)?)?;
                    let last = u32::try_from(last).ok()?;
                    if !bits.contains_range(first..=last) {
                        return None;
                    }
                    IntervalSelection::Range { first, last }
                }
            };
            *slots.get_mut(slot_index(first_key, key))? = certificate;
        }
        Some(Self {
            mask,
            first_key,
            slots,
        })
    }

    /// Read the original immutable mask, including its original representation.
    pub fn mask(&self) -> &RowAddrMask {
        self.mask.as_ref()
    }

    /// Whether this partition's lower hint admits the original dense policy.
    #[inline]
    pub fn is_dense_selection_eligible(&self, input_lower_bound: usize) -> bool {
        input_lower_bound >= self.slots.len()
    }

    /// Test membership using only exact certificates from the original mask.
    #[inline]
    pub fn selected(&self, row_id: u64) -> bool {
        let address = RowAddress::new_from_u64(row_id);
        match self
            .slots
            .get(slot_index(self.first_key, address.fragment_id()))
        {
            Some(IntervalSelection::Full) => true,
            Some(IntervalSelection::Range { first, last }) => {
                let offset = address.row_offset();
                *first <= offset && offset <= *last
            }
            Some(IntervalSelection::Missing) | None => false,
        }
    }

    /// Visit selected positions in input order, preserving duplicate row IDs.
    ///
    /// Short or unknown iterator hints keep the original mask membership path.
    pub fn for_each_selected<'a>(
        &self,
        row_ids: impl Iterator<Item = &'a u64>,
        mut visit: impl FnMut(usize, u64),
    ) {
        if !self.is_dense_selection_eligible(row_ids.size_hint().0) {
            for (position, &row_id) in row_ids.enumerate() {
                if self.mask.selected(row_id) {
                    visit(position, row_id);
                }
            }
            return;
        }
        for (position, &row_id) in row_ids.enumerate() {
            if self.selected(row_id) {
                visit(position, row_id);
            }
        }
    }

    /// Return selected iterator positions, with the original u64 position API.
    pub fn selected_indices<'a>(&self, row_ids: impl Iterator<Item = &'a u64>) -> Vec<u64> {
        let mut selected = Vec::new();
        self.for_each_selected(row_ids, |position, _| selected.push(position as u64));
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::super::{NullableRowAddrMask, NullableRowAddrSet};
    use super::*;
    use std::cell::Cell;

    fn row(key: u32, offset: u32) -> u64 {
        RowAddress::new_from_parts(key, offset).into()
    }
    fn ranges(first: u32, last: u32) -> RowAddrMask {
        RowAddrMask::from_allowed(
            (0..32)
                .flat_map(|key| (first..=last).map(move |offset| row(key, offset)))
                .collect(),
        )
    }

    /// Compaction writes new fragments above every existing key. The view must
    /// size its table by the active key span, not by the largest key.
    #[test]
    fn prepared_membership_uses_active_fragment_span() {
        for (first, count) in [(1466_u32, 702_u32), (u32::MAX - 31, 32)] {
            let mask = RowAddrMask::from_allowed(
                (0..count)
                    .flat_map(|slot| (3..=5).map(move |offset| row(first + slot, offset)))
                    .collect(),
            );
            let prepared = PreparedRowAddrMask::try_new(Arc::new(mask), [1 << 20]).unwrap();
            assert!(prepared.is_dense_selection_eligible(count as usize));
            assert!(!prepared.is_dense_selection_eligible(count as usize - 1));
            for slot in [0, count / 2, count - 1] {
                assert!(prepared.selected(row(first + slot, 4)));
                assert!(!prepared.selected(row(first + slot, 2)));
                assert!(!prepared.selected(row(first + slot, 6)));
            }
            assert!(!prepared.selected(row(first - 1, 4)));
            assert!(!prepared.selected(row(0, 4)));
            if let Some(next) = (first + (count - 1)).checked_add(1) {
                assert!(!prepared.selected(row(next, 4)));
            }
            let rows = [
                row(first, 4),
                row(0, 4),
                row(first + (count - 1), 3),
                row(first, 4),
            ];
            assert_eq!(prepared.selected_indices(rows.iter()), [0, 2, 3]);
        }
    }

    #[test]
    fn prepared_membership_exact_snapshot_and_positions() {
        for (first, last) in [(7, 39), (0, 8192), (u32::MAX - 32, u32::MAX)] {
            let mut mask = Arc::new(ranges(first, last));
            let original = mask.clone();
            let proof = PreparedRowAddrMask::prepare(mask.clone(), [usize::MAX]).unwrap();
            *Arc::make_mut(&mut mask) = RowAddrMask::allow_nothing();
            let prepared = proof().unwrap();
            let mut rows = Vec::new();
            for key in (0..34).rev() {
                rows.extend([row(key, first), row(key, last), row(key, first)]);
                rows.extend([
                    row(key, first.saturating_sub(1)),
                    row(key, last.saturating_add(1)),
                ]);
            }
            assert_eq!(
                prepared.selected_indices(rows.iter()),
                original.selected_indices(rows.iter())
            );
            for &id in &rows {
                assert_eq!(prepared.selected(id), original.selected(id));
            }
            assert!(prepared.selected(row(0, first)));
            assert!(!mask.selected(row(0, first)));
        }
    }

    #[test]
    fn prepared_membership_holes_nulls_and_budget_keep_fallback() {
        let mask = Arc::new(ranges(0, 8192));
        assert!(PreparedRowAddrMask::prepare(mask.clone(), [4096]).is_none());
        let holed = Arc::new(
            mask.as_ref()
                .clone()
                .also_block([row(3, 100)].into_iter().collect()),
        );
        assert!(PreparedRowAddrMask::try_new(holed.clone(), [usize::MAX]).is_none());
        assert!(!holed.selected(row(3, 100)));
        let edge = Arc::new(
            mask.as_ref()
                .clone()
                .also_block([row(3, 0)].into_iter().collect()),
        );
        let prepared = PreparedRowAddrMask::try_new(edge, [usize::MAX]).unwrap();
        assert!(!prepared.selected(row(3, 0)));
        assert!(prepared.selected(row(3, 1)));
        let RowAddrMask::AllowList(selected) = ranges(0, 32) else {
            unreachable!()
        };
        let nulls = [row(3, 7)].into_iter().collect();
        let mask = Arc::new(
            NullableRowAddrMask::AllowList(NullableRowAddrSet::new(selected, nulls)).drop_nulls(),
        );
        assert!(PreparedRowAddrMask::try_new(mask.clone(), [4096]).is_none());
        assert!(!mask.selected(row(3, 7)));
    }

    #[test]
    fn prepared_membership_unsupported_shapes_do_not_visit_metadata() {
        let mut full = RowAddrTreeMap::new();
        for key in 0..32 {
            full.insert_fragment(key);
        }
        for mask in [
            RowAddrMask::all_rows(),
            RowAddrMask::allow_nothing(),
            RowAddrMask::from_block(RowAddrTreeMap::new()),
            RowAddrMask::from_allowed(full),
            RowAddrMask::from_allowed((0..8).map(|key| row(key, 0)).collect()),
            RowAddrMask::from_allowed((0..32).map(|key| row(key * 1000, 0)).collect()),
        ] {
            let counts = std::iter::from_fn(|| -> Option<usize> {
                panic!("unsupported shape must not visit metadata")
            });
            assert!(PreparedRowAddrMask::prepare(Arc::new(mask), counts).is_none());
        }
        assert!(
            PreparedRowAddrMask::try_new(Arc::new(ranges(3, 5)), [usize::MAX, usize::MAX])
                .is_none()
        );
    }

    #[test]
    fn prepared_membership_short_and_unknown_hints_keep_original_selection() {
        let mask = Arc::new(ranges(3, 5));
        let prepared = PreparedRowAddrMask::try_new(mask.clone(), [4096]).unwrap();
        let rows = [row(1, 3), row(1, 2), row(1, 3)];
        assert!(!prepared.is_dense_selection_eligible(rows.len()));
        assert_eq!(
            prepared.selected_indices(rows.iter()),
            mask.selected_indices(rows.iter())
        );
        assert_eq!(
            prepared.selected_indices(rows.iter().filter(|_| true)),
            [0, 2]
        );
    }

    #[test]
    fn prepared_membership_small_budget_stops_metadata_charging() {
        let visited = Cell::new(0);
        let sizes = (0..1466).map(|_| {
            visited.set(visited.get() + 1);
            8208
        });
        assert!(!proof_fits_budget(1466, 4434, sizes));
        assert_eq!(visited.get(), 3);
    }
}
