// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Strict priority admission for layered index planes, sharing the cache byte budget.

use super::InternalCacheKey;
use std::collections::{BTreeMap, HashMap};

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
        let mut dropped = Vec::new();
        if let Some(old) = self.remove(&key) {
            dropped.push(old);
        }
        if bytes > capacity {
            dropped.push(value);
            return dropped;
        }
        self.sequence += 1;
        self.entries
            .insert(key, (priority, self.sequence, bytes, value));
        self.order.insert((priority, self.sequence), key);
        self.bytes += bytes as u128;
        while self.bytes > capacity as u128 {
            if let Some(key) = self.order.first_key_value().map(|(_, key)| *key)
                && let Some(value) = self.remove(&key)
            {
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
}
