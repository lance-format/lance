// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::{CacheBackend, InternalCacheKey};
use crate::{Error, Result};

/// Identifies a cached index entry's logical collection and physical segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheEntryOwner {
    /// Digest of the table, branch, and logical index name.
    pub collection: InternalCacheKey,
    /// Physical index segment ID.
    pub segment: InternalCacheKey,
    /// Dataset version that supplied the entry.
    pub version: u64,
}

/// Tracks whether a physical index file has been opened through multiple collections.
#[doc(hidden)]
#[derive(Debug)]
pub struct IndexFileRegistration {
    pub(super) key: InternalCacheKey,
    collection: Option<InternalCacheKey>,
    shared: AtomicBool,
}

impl IndexFileRegistration {
    pub(super) fn new(key: InternalCacheKey, collection: Option<InternalCacheKey>) -> Self {
        Self {
            key,
            collection,
            shared: AtomicBool::new(collection.is_none()),
        }
    }

    pub(super) fn observe(&self, collection: Option<InternalCacheKey>) {
        if self.collection != collection {
            self.shared.store(true, Ordering::Relaxed);
        }
    }

    pub(super) fn is_shared(&self) -> bool {
        self.shared.load(Ordering::Relaxed)
    }
}

/// Defines which old index entries an explicit prewarm may reclaim.
#[derive(Debug)]
pub struct CachePrewarmPolicy {
    /// Digest of the table, branch, and logical index name.
    pub collection: InternalCacheKey,
    /// Allow reclamation before this version only. Use `None` for historical snapshots.
    pub reclaim_before: Option<u64>,
    /// Index segments still referenced by the requested snapshot.
    pub protected_segments: HashSet<InternalCacheKey>,
}

impl CachePrewarmPolicy {
    pub(super) fn can_reclaim(&self, owner: CacheEntryOwner) -> bool {
        self.reclaim_before
            .is_some_and(|version| owner.version < version)
            && owner.collection == self.collection
            && !self.protected_segments.contains(&owner.segment)
    }
}

#[derive(Default)]
struct Observations {
    resident: HashSet<InternalCacheKey>,
    error: Option<String>,
}

pub(super) struct ScopedPrewarm {
    backend: Arc<dyn CacheBackend>,
    pub(super) policy: CachePrewarmPolicy,
    observations: Mutex<Observations>,
}

tokio::task_local! {
    pub(super) static PREWARM: Arc<ScopedPrewarm>;
}

impl ScopedPrewarm {
    pub(super) fn new(backend: Arc<dyn CacheBackend>, policy: CachePrewarmPolicy) -> Self {
        Self {
            backend,
            policy,
            observations: Mutex::default(),
        }
    }

    pub(super) fn current(backend: &Arc<dyn CacheBackend>) -> Option<Arc<Self>> {
        PREWARM
            .try_with(|scope| Arc::ptr_eq(backend, &scope.backend).then(|| scope.clone()))
            .ok()
            .flatten()
    }

    pub(super) fn record_resident(&self, key: InternalCacheKey) {
        self.observations
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .resident
            .insert(key);
    }

    pub(super) fn record_error(&self, error: Error) {
        self.observations
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .error
            .get_or_insert_with(|| error.to_string());
    }

    pub(super) fn check_error(&self) -> Result<()> {
        let observed = self
            .observations
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        match &observed.error {
            Some(error) => Err(Error::invalid_input(error.clone())),
            None => Ok(()),
        }
    }

    pub(super) async fn verify(&self) -> Result<()> {
        self.check_error()?;
        let keys = {
            let mut observed = self
                .observations
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            std::mem::take(&mut observed.resident)
        };
        for key in keys {
            if !self.backend.peek_resident(&key).await {
                let capacity = self.backend.capacity_bytes().unwrap_or_default();
                return Err(Error::invalid_input(format!(
                    "index cache prewarm lost a resident entry during loading (cache capacity {capacity} bytes)"
                )));
            }
        }
        Ok(())
    }
}
