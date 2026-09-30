// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Strict priority admission for layered index planes, sharing the cache byte budget.

use super::InternalCacheKey;
use std::collections::{BTreeMap, HashMap};

/// Priority of a pinned-kind entry (a value admitted with a
/// [`CachePin`](super::CachePin)): above sign planes (3), so an idle pinned
/// entry is evicted after every plane. While leased within its budget it is
/// never evicted at all.
pub const PINNED_PRIORITY: u8 = 4;

pub struct PriorityEntries<T> {
    entries: HashMap<InternalCacheKey, (u8, u64, usize, T)>,
    order: BTreeMap<(u8, u64), InternalCacheKey>,
    sequence: u64,
    bytes: u128,
}

impl<T> Default for PriorityEntries<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            order: BTreeMap::new(),
            sequence: 0,
            bytes: 0,
        }
    }
}

impl<T: Clone> PriorityEntries<T> {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn bytes(&self) -> usize {
        self.bytes as usize
    }
    pub fn get(&mut self, key: &InternalCacheKey) -> Option<T> {
        let (priority, stamp, _, value) = self.entries.get_mut(key)?;
        self.order.remove(&(*priority, *stamp));
        self.sequence += 1;
        *stamp = self.sequence;
        self.order.insert((*priority, *stamp), *key);
        Some(value.clone())
    }
    /// Look up an entry without refreshing its recency.
    pub fn peek(&self, key: &InternalCacheKey) -> Option<&T> {
        self.entries.get(key).map(|(_, _, _, value)| value)
    }

    /// Membership only; unlike [`get`](Self::get), recency is unchanged.
    pub fn contains(&self, key: &InternalCacheKey) -> bool {
        self.entries.contains_key(key)
    }
    pub fn remove(&mut self, key: &InternalCacheKey) -> Option<T> {
        let (priority, stamp, bytes, value) = self.entries.remove(key)?;
        self.order.remove(&(priority, stamp));
        self.bytes -= bytes as u128;
        Some(value)
    }
    pub fn insert(
        &mut self,
        key: InternalCacheKey,
        value: T,
        bytes: usize,
        priority: u8,
        capacity: usize,
    ) -> Vec<T> {
        self.insert_with(key, value, bytes, priority, capacity, |_| false)
    }

    /// [`insert`](Self::insert) that never evicts an entry `is_pinned`
    /// reports pinned, the inserted one included: pinned entries stay even
    /// when only they are left over capacity. The pin budget keeps them to a
    /// share of the capacity.
    pub fn insert_with(
        &mut self,
        key: InternalCacheKey,
        value: T,
        bytes: usize,
        priority: u8,
        capacity: usize,
        is_pinned: impl Fn(&T) -> bool,
    ) -> Vec<T> {
        let mut dropped = Vec::new();
        if let Some(old) = self.remove(&key) {
            dropped.push(old);
        }
        if bytes > capacity && !is_pinned(&value) {
            dropped.push(value);
            return dropped;
        }
        self.sequence += 1;
        self.entries
            .insert(key, (priority, self.sequence, bytes, value));
        self.order.insert((priority, self.sequence), key);
        self.bytes += bytes as u128;
        while self.bytes > capacity as u128 {
            let victim = self
                .order
                .values()
                .find(|key| {
                    self.entries
                        .get(key)
                        .is_some_and(|(_, _, _, value)| !is_pinned(value))
                })
                .copied();
            let Some(victim) = victim else {
                break;
            };
            if let Some(value) = self.remove(&victim) {
                dropped.push(value);
            }
        }
        dropped
    }
    pub fn clear(&mut self) -> Vec<T> {
        self.order.clear();
        self.bytes = 0;
        self.entries
            .drain()
            .map(|(_, (_, _, _, value))| value)
            .collect()
    }
    pub fn snapshot(&self) -> Vec<(InternalCacheKey, usize, T)> {
        self.entries
            .iter()
            .map(|(key, (_, _, bytes, value))| (*key, *bytes, value.clone()))
            .collect()
    }
    pub fn stats(&self) -> [(usize, usize); 3] {
        let mut stats = [(0, 0); 3];
        for (priority, _, bytes, _) in self.entries.values() {
            if (1..=3).contains(priority) {
                let s = &mut stats[(3 - priority) as usize];
                s.0 += 1;
                s.1 += bytes;
            }
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_preserves_plane_priority_and_budget() {
        let mut cache = PriorityEntries::default();
        let sign = InternalCacheKey::from_bytes([1; 16]);
        let high = InternalCacheKey::from_bytes([2; 16]);
        let low = InternalCacheKey::from_bytes([3; 16]);
        cache.insert(low, 1, 30, 1, 100);
        cache.insert(high, 2, 50, 2, 100);
        assert_eq!(cache.insert(sign, 3, 40, 3, 100), vec![1]);
        assert_eq!(cache.get(&sign), Some(3));
        assert_eq!(cache.get(&high), Some(2));
        assert_eq!(cache.get(&low), None);
        assert_eq!(cache.insert(low, 4, 30, 1, 100), vec![4]);
        assert_eq!(cache.bytes(), 90);
        assert_eq!(cache.stats(), [(1, 40), (1, 50), (0, 0)]);
        assert_eq!(
            cache.insert(InternalCacheKey::from_bytes([4; 16]), 5, 60, 3, 100),
            vec![2]
        );
        assert_eq!(cache.bytes(), 100);
    }

    #[test]
    fn contains_reports_self_evicted_admissions() {
        let (sign, low) = (
            InternalCacheKey::from_bytes([1; 16]),
            InternalCacheKey::from_bytes([3; 16]),
        );
        let mut cache = PriorityEntries::default();
        cache.insert(sign, 1, 80, 3, 100);
        assert!(cache.contains(&sign));
        // The low plane is the minimum of the strict order, so its own
        // admission evicts it and the sign plane stays resident.
        assert_eq!(cache.insert(low, 2, 40, 1, 100), vec![2]);
        assert!(!cache.contains(&low));
        assert!(cache.contains(&sign));
    }

    #[test]
    fn peek_does_not_refresh_recency() {
        let (a, b, c) = (
            InternalCacheKey::from_bytes([1; 16]),
            InternalCacheKey::from_bytes([2; 16]),
            InternalCacheKey::from_bytes([3; 16]),
        );
        let mut peeked = PriorityEntries::default();
        peeked.insert(a, 1, 40, 1, 100);
        peeked.insert(b, 2, 40, 1, 100);
        assert_eq!(peeked.peek(&a), Some(&1));
        assert!(peeked.contains(&a));
        assert_eq!(peeked.insert(c, 3, 40, 1, 100), vec![1]);
        assert!(!peeked.contains(&a) && peeked.peek(&a).is_none());
        assert!(peeked.contains(&b) && peeked.contains(&c));

        // A real access protects the same entry, so the test observes recency.
        let mut touched = PriorityEntries::default();
        touched.insert(a, 1, 40, 1, 100);
        touched.insert(b, 2, 40, 1, 100);
        assert_eq!(touched.get(&a), Some(1));
        assert_eq!(touched.insert(c, 3, 40, 1, 100), vec![2]);
        assert!(touched.contains(&a));
    }

    /// Eviction passes over pinned entries, however old or low their
    /// priority, and a pinned entry heavier than the budget is admitted; only
    /// pinned entries can hold the budget over capacity.
    #[test]
    fn eviction_skips_pinned_entries() {
        let key = |id| InternalCacheKey::from_bytes([id; 16]);
        let is_pinned = |value: &(u32, bool)| value.1;
        let mut cache = PriorityEntries::default();
        cache.insert_with(key(1), (1, true), 40, 1, 100, is_pinned);
        cache.insert_with(key(2), (2, false), 40, 3, 100, is_pinned);
        // The oldest, lowest entry is pinned: the sign entry goes instead.
        let dropped = cache.insert_with(key(3), (3, false), 40, 3, 100, is_pinned);
        assert_eq!(dropped, vec![(2, false)]);
        assert!(cache.contains(&key(1)) && cache.contains(&key(3)));
        // Nothing unpinned is left to evict but the new entry itself.
        let dropped = cache.insert_with(key(4), (4, true), 120, 1, 100, is_pinned);
        assert_eq!(dropped, vec![(3, false)]);
        assert_eq!(cache.bytes(), 160);
        assert!(cache.contains(&key(1)) && cache.contains(&key(4)));
        // Once unpinned, an entry heavier than the budget is refused.
        let dropped = cache.insert_with(key(5), (5, false), 120, 4, 100, is_pinned);
        assert_eq!(dropped, vec![(5, false)]);
    }

    /// An idle pinned-kind entry outranks sign planes, so a sign plane is
    /// evicted first; among pinned-kind entries the oldest goes first.
    #[test]
    fn pinned_priority_outranks_sign() {
        let key = |id| InternalCacheKey::from_bytes([id; 16]);
        let mut cache = PriorityEntries::default();
        cache.insert(key(1), 1, 40, PINNED_PRIORITY, 100);
        cache.insert(key(2), 2, 40, 3, 100);
        assert_eq!(cache.insert(key(3), 3, 40, 3, 100), vec![2]);
        assert_eq!(cache.insert(key(4), 4, 40, PINNED_PRIORITY, 100), vec![3]);
        assert_eq!(cache.insert(key(5), 5, 40, PINNED_PRIORITY, 100), vec![1]);
        // Pinned-kind entries are not plane statistics.
        assert_eq!(cache.stats(), [(0, 0); 3]);
    }
}
