// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One read per leaf for concurrent walks over many versions of a table.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use futures::channel::oneshot;
use lance_core::Result;
use lance_core::cache::LanceCache;

use super::{CheckedLeaf, LeafKey, LeafRead, NodeStore, Sighted, Sighting};
use crate::format::pb;

/// Checked leaves shared by concurrent walks over many versions of one table,
/// such as the manifest walks of one cleanup run. Versions share most leaves,
/// so the walks often miss the same leaf at once. A walk that misses a leaf
/// another walk is already reading waits for that read instead of issuing its
/// own, and its wait counts as the leaf's second sighting.
///
/// Every read it serves runs in a spawned task, so a lead read always makes
/// progress. A walk that held a lead across a loop it might stop polling could
/// leave its followers waiting, which is why only
/// [`FragmentTree::leaf_stream`](crate::fragment_metadata::FragmentTree::leaf_stream)
/// reads through it.
///
/// ```
/// # use std::sync::Arc;
/// # use lance_core::cache::LanceCache;
/// # use lance_table::fragment_metadata::FragmentTree;
/// # use lance_table::fragment_metadata::SingleFlightLeafCache;
/// # use futures::TryStreamExt;
/// # async fn example(versions: Vec<FragmentTree>) -> lance_core::Result<()> {
/// let leaves = SingleFlightLeafCache::new(LanceCache::with_capacity(64 * 1024 * 1024));
/// let walks = versions.into_iter().map(|tree| {
///     Arc::new(tree).leaf_stream(&leaves, |_| true).try_collect::<Vec<_>>()
/// });
/// futures::future::try_join_all(walks).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct SingleFlightLeafCache {
    leaves: LanceCache,
    reads: Arc<ReadsInFlight>,
}

impl SingleFlightLeafCache {
    pub fn new(leaves: LanceCache) -> Self {
        Self {
            leaves,
            reads: Arc::default(),
        }
    }

    /// The cache that holds the checked leaves, which interior reads may share.
    pub fn leaves(&self) -> &LanceCache {
        &self.leaves
    }

    #[cfg(test)]
    pub(in crate::fragment_metadata) fn reads_in_flight(&self) -> usize {
        self.reads.lock().len()
    }

    #[cfg(test)]
    pub(in crate::fragment_metadata) fn waiting_readers(&self) -> usize {
        self.reads.lock().values().map(Vec::len).sum()
    }
}

impl std::fmt::Debug for SingleFlightLeafCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleFlightLeafCache")
            .field("leaves", &self.leaves)
            .field("reads_in_flight", &self.reads.lock().len())
            .finish()
    }
}

type Waiters = Vec<oneshot::Sender<Arc<CheckedLeaf>>>;

/// The readers waiting on each leaf read in flight.
#[derive(Default)]
struct ReadsInFlight(Mutex<BTreeMap<LeafKey, Waiters>>);

impl ReadsInFlight {
    fn lock(&self) -> MutexGuard<'_, BTreeMap<LeafKey, Waiters>> {
        // No critical section leaves the map half updated, so a panic in one
        // reader must not stop the others.
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn join(&self, key: LeafKey) -> Joined<'_> {
        let mut reads = self.lock();
        if let Some(waiters) = reads.get_mut(&key) {
            let (landed, receiver) = oneshot::channel();
            waiters.push(landed);
            return Joined::Wait(receiver);
        }
        reads.insert(key.clone(), Vec::new());
        Joined::Lead(LeadRead {
            reads: self,
            key,
            landed: false,
        })
    }
}

enum Joined<'a> {
    Lead(LeadRead<'a>),
    Wait(oneshot::Receiver<Arc<CheckedLeaf>>),
}

/// The one read of a leaf that concurrent misses wait on. Dropping it before
/// it lands, on an error or when its task is aborted, drops the waiters'
/// senders, and each waiter tries the read again.
struct LeadRead<'a> {
    reads: &'a ReadsInFlight,
    key: LeafKey,
    landed: bool,
}

impl LeadRead<'_> {
    /// End the read with no waiter to serve, or hand the lead back when a
    /// reader joined. Checking and removing under one lock means no reader
    /// can join a read that has already ended.
    fn land_alone(mut self) -> std::result::Result<(), Self> {
        let mut reads = self.reads.lock();
        if reads
            .get(&self.key)
            .is_some_and(|waiters| !waiters.is_empty())
        {
            drop(reads);
            return Err(self);
        }
        reads.remove(&self.key);
        drop(reads);
        self.landed = true;
        Ok(())
    }

    /// End the read and serve every waiter. The caller stores the leaf in the
    /// cache first, so a reader that arrives after the read ends finds it.
    fn land(mut self, leaf: &Arc<CheckedLeaf>) {
        let waiters = self.reads.lock().remove(&self.key).unwrap_or_default();
        self.landed = true;
        for waiter in waiters {
            // A waiter whose walk was dropped no longer needs the leaf.
            let _ = waiter.send(leaf.clone());
        }
    }
}

impl Drop for LeadRead<'_> {
    fn drop(&mut self) {
        if !self.landed {
            self.reads.lock().remove(&self.key);
        }
    }
}

impl NodeStore {
    /// A leaf checked against its parent reference and against the end of
    /// its routing range, read through `leaves` with at most one read in
    /// flight per leaf. A leaf read by one walk is sighted, and admitted once
    /// a second walk reads or waits for it, as [`NodeStore::read_leaf_shared`]
    /// admits on a second read.
    pub(in crate::fragment_metadata) async fn read_leaf_single_flight(
        &self,
        leaves: &SingleFlightLeafCache,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<LeafRead> {
        let key = LeafKey::new(self, child)?;
        let read = loop {
            if let Some(leaf) = leaves.leaves.get_with_key(&key).await {
                break LeafRead::Cached(leaf);
            }
            match leaves.reads.join(key.clone()) {
                Joined::Wait(landed) => match landed.await {
                    Ok(leaf) => break LeafRead::Cached(leaf),
                    // The lead failed or was dropped. Sharing its error would
                    // turn it into `Error::Cloned`, so this reader tries again
                    // and reports the error of a read it led itself.
                    Err(oneshot::Canceled) => continue,
                },
                Joined::Lead(lead) => {
                    break self
                        .lead_leaf_read(&leaves.leaves, key, lead, child)
                        .await?;
                }
            }
        };
        crate::fragment_metadata::validation::leaf_range(
            child,
            read.fragments().last().map(|fragment| fragment.id),
            end,
        )?;
        Ok(read)
    }

    async fn lead_leaf_read(
        &self,
        leaves: &LanceCache,
        key: LeafKey,
        lead: LeadRead<'_>,
        child: &pb::FragmentTreeChild,
    ) -> Result<LeafRead> {
        // A read of this leaf may have landed between the miss and the join.
        if let Some(leaf) = leaves.get_with_key(&key).await {
            lead.land(&leaf);
            return Ok(LeafRead::Cached(leaf));
        }
        let fragments = self.decode_leaf(child, self.leaf_object(child)?).await?;
        let sighting = Sighting(&key);
        let lead = if leaves.get_with_key(&sighting).await.is_some() {
            lead
        } else {
            leaves.insert_with_key(&sighting, Arc::new(Sighted)).await;
            match lead.land_alone() {
                // Avoid copying fragments for leaves read only once.
                Ok(()) => return Ok(LeafRead::Owned(fragments)),
                Err(lead) => lead,
            }
        };
        let leaf = Arc::new(CheckedLeaf::new(fragments));
        leaves.insert_with_key(&key, leaf.clone()).await;
        lead.land(&leaf);
        Ok(LeafRead::Cached(leaf))
    }
}
