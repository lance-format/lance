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
//! of an unleased entry, or its next admission, tries again.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Share of each budget partition that pinned entries may hold. The rest
/// always holds evictable weight, so an admission never finds only pinned
/// entries to evict.
pub const PINNED_CAP_FRACTION: f64 = 0.5;

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
            partition_cap: (share as f64 * PINNED_CAP_FRACTION) as u64,
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

/// Where a pin's entry is admitted: the budget of the backend holding it,
/// the partition (shard) of its key and the bytes the backend charges.
#[derive(Debug, Clone)]
struct PinBinding {
    budget: Arc<PinBudget>,
    partition: usize,
    bytes: u64,
}

impl PinBinding {
    fn same_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.budget, &other.budget)
            && self.partition == other.partition
            && self.bytes == other.bytes
    }
}

#[derive(Debug, Default)]
struct PinState {
    /// Live leases.
    holders: usize,
    /// Live admissions of the value in its backend.
    records: usize,
    binding: Option<PinBinding>,
    /// Whether the binding's budget holds the entry's bytes.
    reserved: bool,
    /// Whether the binding's budget counts the entry as leased.
    counted_leased: bool,
}

/// The pin of one cached value; see the [module docs](self). Leases and
/// admissions share it, so every admission of the value is pinned by the
/// same leases: a value evicted while leased (an overflow) and admitted again
/// is pinned again at once.
#[derive(Debug, Default)]
pub struct CachePin {
    state: Mutex<PinState>,
    /// Mirror of `state.reserved`, which eviction reads without the lock.
    pinned: AtomicBool,
}

impl CachePin {
    /// A new pin, with neither leases nor admissions.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Take a lease: the value's entry stays in RAM, on a backend that pins,
    /// until the lease and its clones drop. The first lease of an admitted
    /// entry reserves its bytes, or counts an overflow.
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
    /// overflow and stays evictable. A pin admitted to another budget, or with
    /// another charge, moves to this one.
    pub fn record(
        pin: &Arc<Self>,
        budget: &Arc<PinBudget>,
        partition: usize,
        bytes: u64,
    ) -> PinRecord {
        let binding = PinBinding {
            budget: budget.clone(),
            partition,
            bytes,
        };
        let mut state = pin.lock();
        if !state
            .binding
            .as_ref()
            .is_some_and(|bound| bound.same_as(&binding))
        {
            Self::unbind(&mut state);
            state.binding = Some(binding);
        }
        state.records += 1;
        pin.sync(&mut state, true);
        PinRecord { pin: pin.clone() }
    }

    /// Whether a backend must keep the entry: admitted, leased, and within
    /// its budget's cap.
    pub fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::Acquire)
    }

    /// Live leases on the value.
    pub fn holders(&self) -> usize {
        self.lock().holders
    }

    /// Whether the entry is admitted and leased but found no room under its
    /// budget's cap, so it stays evictable.
    pub fn is_overflowed(&self) -> bool {
        let state = self.lock();
        state.holders > 0 && state.records > 0 && state.binding.is_some() && !state.reserved
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

    fn drop_record(&self) {
        let mut state = self.lock();
        state.records = state.records.saturating_sub(1);
        if state.records == 0 {
            self.sync(&mut state, false);
        }
    }

    /// Release what the binding's budget holds for the entry.
    fn unbind(state: &mut PinState) {
        if let Some(binding) = &state.binding {
            if state.counted_leased {
                binding.budget.count_leased(binding.bytes, false);
            }
            if state.reserved {
                binding.budget.release(binding.partition, binding.bytes);
            }
        }
        state.counted_leased = false;
        state.reserved = false;
    }

    /// Bring the budget in line with the holders and admissions: count a
    /// leased admitted entry as leased, release it once unleased or no
    /// longer admitted, and, when `reserve`, reserve its bytes if it is not
    /// yet pinned.
    fn sync(&self, state: &mut PinState, reserve: bool) {
        let active = state.holders > 0 && state.records > 0;
        match state.binding.clone() {
            Some(binding) if active => {
                if !state.counted_leased {
                    binding.budget.count_leased(binding.bytes, true);
                    state.counted_leased = true;
                }
                if reserve && !state.reserved {
                    state.reserved = binding.budget.try_reserve(binding.partition, binding.bytes);
                }
            }
            _ => Self::unbind(state),
        }
        self.pinned.store(state.reserved, Ordering::Release);
    }
}

/// One admission of a pinned value in a backend; see [`CachePin::record`].
/// The backend keeps it with the entry, and every copy of the entry shares
/// it, so the admission ends when the last copy leaves the backend: evicted,
/// removed, replaced or cleared.
#[derive(Debug)]
pub struct PinRecord {
    pin: Arc<CachePin>,
}

impl PinRecord {
    /// Whether the backend must keep the entry; see [`CachePin::is_pinned`].
    pub fn is_pinned(&self) -> bool {
        self.pin.is_pinned()
    }

    /// The pin of the recorded value.
    pub fn pin(&self) -> &Arc<CachePin> {
        &self.pin
    }
}

impl Drop for PinRecord {
    fn drop(&mut self) {
        self.pin.drop_record();
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

    /// Whether the leased entry is pinned; see [`CachePin::is_pinned`].
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

    #[test]
    fn admission_to_another_budget_moves_the_pin() {
        let (first, second) = (
            Arc::new(PinBudget::new(1000, 1)),
            Arc::new(PinBudget::new(1000, 1)),
        );
        let pin = CachePin::new();
        let _lease = CachePin::lease(&pin);
        let old = CachePin::record(&pin, &first, 0, 100);
        assert_eq!(first.stats().pinned_bytes, 100);
        let _new = CachePin::record(&pin, &second, 0, 100);
        assert_eq!(first.stats().pinned_bytes, 0);
        assert_eq!(second.stats().pinned_bytes, 100);
        drop(old);
        // The pin is still admitted once, so it stays pinned.
        assert!(pin.is_pinned());
    }
}
