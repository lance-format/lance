// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Pinned cache entries: values that stay in RAM while their users hold them.
//!
//! A [`CachePin`] belongs to one cached value ([`PinnedValue`]), and every
//! [`CacheLease`] on it is a holder that uses the value. A backend that pins
//! entries records each admission of the value in its [`PinBudget`]
//! ([`PinRecord`]), and does not evict the entry while it is leased and the
//! budget holds its bytes. An entry nobody leases is evictable like any
//! other.
//!
//! The budget caps what pins hold at [`PINNED_CAP_FRACTION`] of each of its
//! partitions (a backend's shards), so eviction always finds unpinned weight
//! to free and pins never push admissions over capacity. A lease that finds
//! no room leaves its entry evictable and counts an overflow; the next lease
//! of an unleased value, its next admission, or the end of another pinned
//! admission of that value in the same budget partition tries again.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Share of each budget partition that pinned entries may hold. The rest
/// always holds evictable weight, so an admission never finds only pinned
/// entries to evict.
pub const PINNED_CAP_FRACTION: f64 = 0.5;

/// Bytes the pinned entries of one budget partition may hold, for a
/// partition whose share of the capacity is `share_bytes`: the cap a lease
/// compares an entry's charged bytes with. A backend whose largest
/// admissible entry is its partition's share
/// ([`CacheBackend::max_entry_bytes`](super::CacheBackend::max_entry_bytes))
/// pins an entry only when it charges at most this much of it.
pub fn pinned_partition_cap(share_bytes: u64) -> u64 {
    (share_bytes as f64 * PINNED_CAP_FRACTION) as u64
}

/// What a [`PinBudget`] holds, as [`PinBudget::stats`] reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PinnedStats {
    /// Bytes pinned entries may hold at most, over every partition.
    pub cap_bytes: u64,
    /// Entries the budget holds pinned: admitted, leased and within the cap.
    pub pinned_entries: u64,
    /// Bytes of those entries.
    pub pinned_bytes: u64,
    /// Admitted entries with at least one lease, pinned or overflowed.
    pub leased_entries: u64,
    /// Bytes of those entries.
    pub leased_bytes: u64,
    /// Leases and admissions of leased entries that found no room under the
    /// cap and left their entry evictable.
    pub overflow: u64,
}

#[derive(Debug, Default)]
struct BudgetState {
    /// Pinned bytes per partition.
    pinned: Vec<u64>,
    pinned_entries: u64,
    leased_entries: u64,
    leased_bytes: u64,
}

/// Bytes a cache backend lets leased entries pin, split into partitions
/// that each cap their pins at [`PINNED_CAP_FRACTION`] of their share of the
/// capacity. A backend whose shards evict independently uses one partition
/// per shard, so that no shard can fill with pinned entries.
#[derive(Debug)]
pub struct PinBudget {
    partition_cap: u64,
    state: Mutex<BudgetState>,
    overflow: AtomicU64,
}

impl PinBudget {
    /// A budget over `capacity` bytes split into `partitions` equal shares
    /// (at least one).
    pub fn new(capacity: u64, partitions: usize) -> Self {
        let partitions = partitions.max(1);
        let share = capacity / partitions as u64;
        Self {
            partition_cap: pinned_partition_cap(share),
            state: Mutex::new(BudgetState {
                pinned: vec![0; partitions],
                ..Default::default()
            }),
            overflow: AtomicU64::new(0),
        }
    }

    /// What the budget holds now, and the overflows it counted.
    pub fn stats(&self) -> PinnedStats {
        let state = self.lock();
        PinnedStats {
            cap_bytes: self.partition_cap * state.pinned.len() as u64,
            pinned_entries: state.pinned_entries,
            pinned_bytes: state.pinned.iter().sum(),
            leased_entries: state.leased_entries,
            leased_bytes: state.leased_bytes,
            overflow: self.overflow.load(Ordering::Relaxed),
        }
    }

    fn lock(&self) -> MutexGuard<'_, BudgetState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Reserve `bytes` of `partition`, or count an overflow when they do not
    /// fit under its cap.
    fn try_reserve(&self, partition: usize, bytes: u64) -> bool {
        let mut state = self.lock();
        let partition_cap = self.partition_cap;
        let reserved = match state.pinned.get_mut(partition) {
            Some(pinned) if pinned.saturating_add(bytes) <= partition_cap => {
                *pinned += bytes;
                true
            }
            _ => false,
        };
        if reserved {
            state.pinned_entries += 1;
        } else {
            self.overflow.fetch_add(1, Ordering::Relaxed);
        }
        reserved
    }

    fn release(&self, partition: usize, bytes: u64) {
        let mut state = self.lock();
        if let Some(pinned) = state.pinned.get_mut(partition) {
            *pinned = pinned.saturating_sub(bytes);
        }
        state.pinned_entries = state.pinned_entries.saturating_sub(1);
    }

    fn count_leased(&self, bytes: u64, leased: bool) {
        let mut state = self.lock();
        if leased {
            state.leased_entries += 1;
            state.leased_bytes += bytes;
        } else {
            state.leased_entries = state.leased_entries.saturating_sub(1);
            state.leased_bytes = state.leased_bytes.saturating_sub(bytes);
        }
    }
}

/// One admission's charge and eviction protection. The owning pin's state
/// lock serializes updates; its record reads the flag without that lock.
#[derive(Debug)]
struct PinAdmission {
    budget: Arc<PinBudget>,
    partition: usize,
    bytes: u64,
    pinned: Arc<AtomicBool>,
    counted_leased: bool,
}

impl PinAdmission {
    fn sync(&mut self, is_leased: bool) {
        if !is_leased {
            self.release();
            return;
        }
        if !self.counted_leased {
            self.budget.count_leased(self.bytes, true);
            self.counted_leased = true;
        }
        if !self.pinned.load(Ordering::Relaxed) {
            let reserved = self.budget.try_reserve(self.partition, self.bytes);
            self.pinned.store(reserved, Ordering::Release);
        }
    }

    fn release(&mut self) {
        // Make the entry evictable before another admission can reuse its
        // budget, so eviction never protects uncharged bytes.
        if self.pinned.swap(false, Ordering::Release) {
            self.budget.release(self.partition, self.bytes);
        }
        if self.counted_leased {
            self.budget.count_leased(self.bytes, false);
            self.counted_leased = false;
        }
    }
}

#[derive(Debug, Default)]
struct PinState {
    /// Live leases.
    holders: usize,
    /// Every live cache admission has its own reservation, including aliases
    /// of the same value under different keys or in different backends.
    admissions: Vec<PinAdmission>,
}

/// The pin of one cached value; see the [module docs](self). Leases and
/// admissions share lease ownership, while each admission independently
/// reserves its own cache charge. A value evicted while leased (an overflow)
/// can be pinned again on its next admission if that budget has room.
#[derive(Debug, Default)]
pub struct CachePin {
    state: Mutex<PinState>,
    /// Whether any admission is pinned. Eviction uses the admission-specific
    /// flag in its `PinRecord`, never this value-wide status.
    pinned: AtomicBool,
}

impl CachePin {
    /// A new pin, with neither leases nor admissions.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Take a lease: the value's entry stays in RAM, on a backend that pins,
    /// until the lease and its clones drop. The first lease of an admitted
    /// value reserves each admission's bytes independently, or counts an
    /// overflow in that admission's budget.
    pub fn lease(pin: &Arc<Self>) -> CacheLease {
        let mut state = pin.lock();
        state.holders += 1;
        if state.holders == 1 {
            pin.sync(&mut state, true);
        }
        CacheLease { pin: pin.clone() }
    }

    /// Record an admission of the value in `budget`, in `partition`, charged
    /// `bytes`. The entry is pinned while the returned record and a lease
    /// live, if the budget has room; a leased entry that finds none counts an
    /// overflow and stays evictable. Each call creates a separate reservation;
    /// admitting the value again never transfers an existing admission's charge.
    pub fn record(
        pin: &Arc<Self>,
        budget: &Arc<PinBudget>,
        partition: usize,
        bytes: u64,
    ) -> PinRecord {
        let pinned = Arc::new(AtomicBool::new(false));
        let mut admission = PinAdmission {
            budget: budget.clone(),
            partition,
            bytes,
            pinned: pinned.clone(),
            counted_leased: false,
        };
        let mut state = pin.lock();
        admission.sync(state.holders > 0);
        state.admissions.push(admission);
        pin.update_pinned(&state);
        PinRecord {
            pin: pin.clone(),
            pinned,
        }
    }

    /// Whether at least one admission is leased and within its budget's cap.
    /// This does not imply that every admission is protected. Backends must
    /// use [`PinRecord::is_pinned`] when deciding whether to evict an entry.
    pub fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::Acquire)
    }

    /// Live leases on the value.
    pub fn holders(&self) -> usize {
        self.lock().holders
    }

    /// Live admissions of the value: the [`PinRecord`]s its backends hold,
    /// one per admission however many copies of the entry share it. A
    /// backend dropping an entry whose record is the only one left drops
    /// the value from the cache; one replaced by another admission of the
    /// same value does not.
    pub fn admissions(&self) -> usize {
        self.lock().admissions.len()
    }

    /// Whether any leased admission found no room under its budget's cap
    /// and remains evictable. Other admissions may still be pinned.
    pub fn is_overflowed(&self) -> bool {
        let state = self.lock();
        state.holders > 0
            && state
                .admissions
                .iter()
                .any(|admission| !admission.pinned.load(Ordering::Relaxed))
    }

    fn lock(&self) -> MutexGuard<'_, PinState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn release_holder(&self) {
        let mut state = self.lock();
        state.holders = state.holders.saturating_sub(1);
        if state.holders == 0 {
            self.sync(&mut state, false);
        }
    }

    fn drop_record(&self, pinned: &Arc<AtomicBool>) {
        let mut state = self.lock();
        let Some(index) = state
            .admissions
            .iter()
            .position(|admission| Arc::ptr_eq(&admission.pinned, pinned))
        else {
            debug_assert!(false, "a live pin record must have an admission");
            return;
        };
        let mut removed = state.admissions.swap_remove(index);
        let is_reserved = removed.pinned.load(Ordering::Relaxed);
        removed.release();
        if is_reserved && state.holders > 0 {
            // A replacement can arrive before the old record drops. Once
            // its charge is released, retry the value's remaining admissions
            // in that partition without disturbing other budgets.
            for admission in &mut state.admissions {
                if Arc::ptr_eq(&admission.budget, &removed.budget)
                    && admission.partition == removed.partition
                    && !admission.pinned.load(Ordering::Relaxed)
                {
                    admission.sync(true);
                }
            }
        }
        self.update_pinned(&state);
    }

    fn sync(&self, state: &mut PinState, is_leased: bool) {
        for admission in &mut state.admissions {
            admission.sync(is_leased);
        }
        self.update_pinned(state);
    }

    fn update_pinned(&self, state: &PinState) {
        self.pinned.store(
            state
                .admissions
                .iter()
                .any(|admission| admission.pinned.load(Ordering::Relaxed)),
            Ordering::Release,
        );
    }
}

/// One admission of a pinned value in a backend; see [`CachePin::record`].
/// The backend keeps it with the entry, and every copy of the entry shares
/// it, so the admission ends when the last copy leaves the backend: evicted,
/// removed, replaced or cleared.
#[derive(Debug)]
pub struct PinRecord {
    pin: Arc<CachePin>,
    pinned: Arc<AtomicBool>,
}

impl PinRecord {
    /// Whether this admission is leased and holds its own budget reservation.
    /// Another admission of the same value cannot provide eviction protection
    /// for this record.
    pub fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::Acquire)
    }

    /// The pin of the recorded value.
    pub fn pin(&self) -> &Arc<CachePin> {
        &self.pin
    }
}

impl Drop for PinRecord {
    fn drop(&mut self) {
        self.pin.drop_record(&self.pinned);
    }
}

/// A holder of a pinned value; see [`CachePin::lease`]. Clones are holders
/// too; the value's entry is evictable again once every lease dropped.
#[derive(Debug)]
pub struct CacheLease {
    pin: Arc<CachePin>,
}

impl CacheLease {
    /// The pin this lease holds.
    pub fn pin(&self) -> &Arc<CachePin> {
        &self.pin
    }

    /// Whether any admission of the leased value is pinned; see
    /// [`CachePin::is_pinned`].
    pub fn is_pinned(&self) -> bool {
        self.pin.is_pinned()
    }
}

impl Clone for CacheLease {
    fn clone(&self) -> Self {
        CachePin::lease(&self.pin)
    }
}

impl Drop for CacheLease {
    fn drop(&mut self) {
        self.pin.release_holder();
    }
}

/// A cache value leased through a pin of its own, so that every admission
/// of the value is pinned by the same leases; see
/// [`LanceCache::get_or_insert_leased_with_key`](super::LanceCache::get_or_insert_leased_with_key).
pub trait PinnedValue {
    /// The value's pin.
    fn cache_pin(&self) -> &Arc<CachePin>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leased_admission_pins_within_the_cap() {
        let budget = Arc::new(PinBudget::new(1000, 1));
        let pin = CachePin::new();
        // A lease before admission pins nothing yet.
        let lease = CachePin::lease(&pin);
        assert!(!pin.is_pinned());
        let record = CachePin::record(&pin, &budget, 0, 400);
        assert!(pin.is_pinned() && lease.is_pinned());
        let stats = budget.stats();
        assert_eq!(
            (stats.cap_bytes, stats.pinned_entries, stats.pinned_bytes),
            (500, 1, 400)
        );
        assert_eq!((stats.leased_entries, stats.leased_bytes), (1, 400));
        // A second holder adds nothing; the last one unpins.
        let clone = lease.clone();
        assert_eq!(pin.holders(), 2);
        drop(lease);
        assert!(pin.is_pinned());
        drop(clone);
        assert!(!pin.is_pinned());
        assert_eq!(budget.stats().pinned_bytes, 0);
        assert_eq!(budget.stats().leased_entries, 0);
        // An admitted entry is pinned by its next first lease.
        let lease = CachePin::lease(&pin);
        assert!(pin.is_pinned());
        drop(record);
        assert!(!lease.is_pinned());
        assert_eq!(
            budget.stats(),
            PinnedStats {
                cap_bytes: 500,
                ..Default::default()
            }
        );
    }

    #[test]
    fn overflow_leaves_the_entry_unpinned_and_retries() {
        let budget = Arc::new(PinBudget::new(1000, 1));
        let (first, second) = (CachePin::new(), CachePin::new());
        let _first_record = CachePin::record(&first, &budget, 0, 300);
        let _second_record = CachePin::record(&second, &budget, 0, 300);
        let first_lease = CachePin::lease(&first);
        let second_lease = CachePin::lease(&second);
        assert!(first.is_pinned());
        assert!(!second.is_pinned() && second.is_overflowed());
        let stats = budget.stats();
        assert_eq!((stats.pinned_bytes, stats.overflow), (300, 1));
        assert_eq!((stats.leased_entries, stats.leased_bytes), (2, 600));
        // Another holder does not retry; the next first lease does.
        let extra = second_lease.clone();
        assert_eq!(budget.stats().overflow, 1);
        drop(first_lease);
        drop((second_lease, extra));
        let _second_lease = CachePin::lease(&second);
        assert!(second.is_pinned());
    }

    #[test]
    fn budget_partitions_cap_independently() {
        let budget = Arc::new(PinBudget::new(1000, 2));
        assert_eq!(budget.stats().cap_bytes, 500);
        let pins = [CachePin::new(), CachePin::new(), CachePin::new()];
        let records: Vec<_> = pins
            .iter()
            .zip([0, 0, 1])
            .map(|(pin, partition)| CachePin::record(pin, &budget, partition, 200))
            .collect();
        let leases: Vec<_> = pins.iter().map(CachePin::lease).collect();
        // 250 bytes per partition: the second entry of partition 0 overflows.
        let pinned: Vec<_> = pins.iter().map(|pin| pin.is_pinned()).collect();
        assert_eq!(pinned, [true, false, true]);
        drop((records, leases));
        assert_eq!(budget.stats().pinned_bytes, 0);
    }

    #[rstest::rstest]
    #[case::same_budget_same_charge(true, 100)]
    #[case::same_budget_different_charge(true, 200)]
    #[case::different_budgets(false, 200)]
    fn admissions_keep_their_own_reservations(
        #[case] is_same_budget: bool,
        #[case] second_bytes: u64,
        #[values(false, true)] is_reverse_drop: bool,
    ) {
        let first = Arc::new(PinBudget::new(1000, 1));
        let second = if is_same_budget {
            first.clone()
        } else {
            Arc::new(PinBudget::new(1000, 1))
        };
        let pin = CachePin::new();
        let lease = CachePin::lease(&pin);
        let first_record = CachePin::record(&pin, &first, 0, 100);
        let second_record = CachePin::record(&pin, &second, 0, second_bytes);
        assert!(first_record.is_pinned() && second_record.is_pinned());
        assert_eq!(pin.admissions(), 2);
        assert_eq!(
            first.stats().pinned_bytes,
            100 + if is_same_budget { second_bytes } else { 0 }
        );
        assert_eq!(
            second.stats().pinned_bytes,
            second_bytes + if is_same_budget { 100 } else { 0 }
        );

        // All admissions share lease ownership, but release their own charges.
        drop(lease);
        assert!(!first_record.is_pinned() && !second_record.is_pinned());
        assert_eq!(first.stats().pinned_bytes, 0);
        assert_eq!(second.stats().leased_bytes, 0);
        let _lease = CachePin::lease(&pin);
        assert!(first_record.is_pinned() && second_record.is_pinned());

        let (removed, kept, removed_budget, kept_budget, kept_bytes) = if is_reverse_drop {
            (second_record, first_record, &second, &first, 100)
        } else {
            (first_record, second_record, &first, &second, second_bytes)
        };
        drop(removed);
        assert_eq!(pin.admissions(), 1);
        assert!(kept.is_pinned() && pin.is_pinned());
        assert_eq!(kept_budget.stats().pinned_bytes, kept_bytes);
        assert_eq!(kept_budget.stats().leased_bytes, kept_bytes);
        assert_eq!(kept_budget.stats().pinned_entries, 1);
        if !is_same_budget {
            assert_eq!(removed_budget.stats().pinned_bytes, 0);
            assert_eq!(removed_budget.stats().leased_entries, 0);
        }
        drop(kept);
        assert_eq!(pin.admissions(), 0);
        assert!(!pin.is_pinned());
        assert_eq!(first.stats().pinned_bytes, 0);
        assert_eq!(second.stats().leased_entries, 0);
    }

    #[test]
    fn admission_overflow_and_retry_stay_in_their_partition() {
        let budget = Arc::new(PinBudget::new(1000, 2));
        let pin = CachePin::new();
        let _lease = CachePin::lease(&pin);
        let first = CachePin::record(&pin, &budget, 0, 200);
        let overflow = CachePin::record(&pin, &budget, 0, 200);
        let other_partition = CachePin::record(&pin, &budget, 1, 200);
        assert!(first.is_pinned() && other_partition.is_pinned());
        assert!(!overflow.is_pinned());
        assert!(pin.is_pinned() && pin.is_overflowed());
        assert_eq!(budget.stats().pinned_bytes, 400);
        assert_eq!(budget.stats().leased_bytes, 600);

        drop(other_partition);
        assert!(first.is_pinned());
        assert!(
            !overflow.is_pinned(),
            "partitions cannot borrow pin capacity"
        );
        drop(first);
        assert!(overflow.is_pinned());
        assert!(!pin.is_overflowed());
        assert_eq!(budget.stats().pinned_entries, 1);
        drop(overflow);
        assert_eq!(
            budget.stats(),
            PinnedStats {
                cap_bytes: 500,
                overflow: 1,
                ..Default::default()
            }
        );
    }

    /// Admissions count records, not the copies of an entry that share one.
    #[test]
    fn admissions_count_records() {
        let budget = Arc::new(PinBudget::new(1000, 1));
        let pin = CachePin::new();
        let _lease = CachePin::lease(&pin);
        assert_eq!(pin.admissions(), 0);
        let record = Arc::new(CachePin::record(&pin, &budget, 0, 100));
        let copy = record.clone();
        assert_eq!(pin.admissions(), 1);
        assert_eq!(budget.stats().pinned_bytes, 100);
        let replacement = CachePin::record(&pin, &budget, 0, 100);
        assert_eq!(pin.admissions(), 2);
        assert_eq!(budget.stats().pinned_bytes, 200);
        drop((record, copy));
        assert_eq!(pin.admissions(), 1);
        assert_eq!(budget.stats().pinned_bytes, 100);
        drop(replacement);
        assert_eq!(pin.admissions(), 0);
        assert_eq!(budget.stats().pinned_bytes, 0);
    }

    #[test]
    fn concurrent_admissions_reserve_and_release_independently() {
        let budget = Arc::new(PinBudget::new(1000, 2));
        let pin = CachePin::new();
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let _lease = CachePin::lease(&pin);
                    let record = CachePin::record(&pin, &budget, 0, 100);
                    barrier.wait();
                    assert!(budget.stats().pinned_bytes <= 250);
                    drop(record);
                    for _ in 0..32 {
                        let record = CachePin::record(&pin, &budget, 0, 100);
                        if record.is_pinned() {
                            assert!(budget.stats().pinned_bytes >= 100);
                        }
                        assert!(budget.stats().pinned_bytes <= 250);
                        std::thread::yield_now();
                    }
                });
            }
        });
        assert_eq!(pin.admissions(), 0);
        assert_eq!(pin.holders(), 0);
        assert_eq!(budget.stats().pinned_bytes, 0);
        assert_eq!(budget.stats().leased_entries, 0);
        assert!(budget.stats().overflow > 0);
    }
}
