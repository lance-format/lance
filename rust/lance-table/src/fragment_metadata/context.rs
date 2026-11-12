// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::cache::LanceCache;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::ScanScheduler;
use object_store::path::Path;

use super::node::FragmentTreeConfig;
use super::store::NodeStore;

/// Storage and the cache shared by snapshots in one dataset namespace.
/// Configure foreign dataset locations before opening a snapshot.
///
/// ```
/// # use std::sync::Arc;
/// # use lance_core::cache::LanceCache;
/// # use lance_io::{object_store::ObjectStore, scheduler::ScanScheduler};
/// # use lance_table::fragment_metadata::TreeContext;
/// # use object_store::path::Path;
/// # fn example(store: Arc<ObjectStore>, base: Path, scheduler: Arc<ScanScheduler>) {
/// let context = TreeContext::new(store, base, scheduler)
///     .with_cache(LanceCache::with_capacity(64 * 1024 * 1024));
/// # }
/// ```
#[derive(Clone)]
pub struct TreeContext {
    object_store: Arc<ObjectStore>,
    base: Path,
    scheduler: Arc<ScanScheduler>,
    cache: Option<LanceCache>,
    foreign_bases: HashMap<u32, (Arc<ObjectStore>, Path)>,
}

impl TreeContext {
    /// Create the context for a dataset's tree objects.
    pub fn new(object_store: Arc<ObjectStore>, base: Path, scheduler: Arc<ScanScheduler>) -> Self {
        Self {
            object_store,
            base,
            scheduler,
            cache: None,
            foreign_bases: HashMap::new(),
        }
    }

    /// Share checked leaves and interior nodes across snapshots.
    pub fn with_cache(mut self, cache: LanceCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Resolve tree objects whose `base_id` names another dataset. Each key is a
    /// `BasePath` id, and each value is the store and root path of that dataset.
    pub fn with_foreign_bases(mut self, bases: HashMap<u32, (Arc<ObjectStore>, Path)>) -> Self {
        self.foreign_bases = bases;
        self
    }

    pub(super) fn into_store(self, config: &FragmentTreeConfig, root_frontier: u64) -> NodeStore {
        let mut store = NodeStore::new(self.object_store, self.base, self.scheduler);
        store.root_frontier = root_frontier;
        store.hard_capacity_bytes = config.hard_capacity_bytes;
        store.set_foreign_bases(self.foreign_bases);
        if let Some(cache) = self.cache {
            store.set_cache(cache);
        }
        store
    }
}
