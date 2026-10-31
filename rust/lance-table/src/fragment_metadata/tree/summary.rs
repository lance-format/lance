// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Version summaries read leaf by leaf, for a caller that summarizes many
//! versions of one table.

use std::collections::BTreeMap;

use futures::{StreamExt, TryFutureExt, TryStreamExt};
use prost::Message;

use super::{BufferedChild, FragmentTree, SpawnedRead};
use crate::format::ManifestSummary;
use crate::fragment_metadata::store::{LeafLocation, NodeStore, StoredFragments};
use lance_core::Result;

/// Leaf summaries that [`FragmentTree::summary`] computed for earlier
/// versions of one table, so each stored leaf is read once. A leaf with routed
/// actions is replayed again unless its actions equal, byte for byte, the
/// ones last replayed there. Summarize versions in order, since a leaf
/// repeats its actions from one version to the next until a commit adds to
/// them. Keeping every replay instead of the last would grow with the square
/// of an append history.
///
/// Nothing is evicted, so hold one for a single pass over versions.
#[derive(Default)]
pub struct LeafSummaries {
    /// A stored leaf's records never change, so its location identifies them.
    stored: BTreeMap<LeafLocation, ManifestSummary>,
    replayed: BTreeMap<LeafLocation, (Vec<u8>, ManifestSummary)>,
}

impl LeafSummaries {
    fn get(&self, leaf: &LeafIdentity) -> Option<&ManifestSummary> {
        if leaf.routed_actions.is_empty() {
            return self.stored.get(&leaf.location);
        }
        let (routed_actions, summary) = self.replayed.get(&leaf.location)?;
        (*routed_actions == leaf.routed_actions).then_some(summary)
    }

    fn insert(&mut self, leaf: LeafIdentity, summary: ManifestSummary) {
        if leaf.routed_actions.is_empty() {
            self.stored.insert(leaf.location, summary);
        } else {
            self.replayed
                .insert(leaf.location, (leaf.routed_actions, summary));
        }
    }
}

/// What fixes a leaf's records in one version. Replay depends only on the
/// stored leaf and the actions routed to it, kept here as their
/// length-delimited encodings in routing order, empty when none route there.
struct LeafIdentity {
    location: LeafLocation,
    routed_actions: Vec<u8>,
}

impl LeafIdentity {
    fn new(store: &NodeStore, (child, actions, _): &BufferedChild) -> Result<Self> {
        Ok(Self {
            location: LeafLocation(store.child_location(child)?),
            routed_actions: actions
                .iter()
                .flat_map(Message::encode_length_delimited_to_vec)
                .collect(),
        })
    }
}

impl FragmentTree {
    /// The summary [`Manifest::summary`](crate::format::Manifest::summary)
    /// reports for this version's fragments, read leaf by leaf without
    /// building the version's inventory. A leaf `leaves` already holds is not
    /// read again. The others are read through this tree's configured leaf
    /// cache, each decoding on its own task, and are added to `leaves`.
    ///
    /// ```no_run
    /// # use lance_table::fragment_metadata::{FragmentTree, LeafSummaries};
    /// # async fn example(versions: Vec<FragmentTree>) -> lance_core::Result<()> {
    /// let mut leaves = LeafSummaries::default();
    /// for tree in versions {
    ///     let summary = tree.summary(&mut leaves).await?;
    ///     assert_eq!(summary.total_fragments, tree.count_fragments());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn summary(&self, leaves: &mut LeafSummaries) -> Result<ManifestSummary> {
        let mut walk = self.start_walk_from(0);
        // A root without children holds its whole table in its buffer, which
        // starting the walk has already replayed. Any other walk starts empty.
        let mut summary = ManifestSummary::of(walk.ready.make_contiguous());
        let mut unread = Vec::new();
        while let Some(leaf) = self.walk_next_leaf(&mut walk).await? {
            let identity = LeafIdentity::new(&self.store, &leaf)?;
            match leaves.get(&identity) {
                Some(known) => summary.add(known),
                None => unread.push((identity, leaf)),
            }
        }
        let reads = unread.into_iter().map(|(identity, leaf)| {
            SpawnedRead(tokio::spawn(summarize_leaf(self.store.clone(), leaf)))
                .map_ok(move |leaf_summary| (identity, leaf_summary))
        });
        let mut reads =
            futures::stream::iter(reads).buffered(self.store.object_store.io_parallelism());
        while let Some((identity, leaf_summary)) = reads.try_next().await? {
            summary.add(&leaf_summary);
            leaves.insert(identity, leaf_summary);
        }
        Ok(summary)
    }
}

async fn summarize_leaf(
    store: NodeStore,
    (child, actions, end): BufferedChild,
) -> Result<ManifestSummary> {
    let read = store.read_leaf_shared(&child, end).await?;
    if actions.is_empty() {
        return Ok(ManifestSummary::of(&StoredFragments(read)));
    }
    Ok(ManifestSummary::of(
        &store.replay(read.into_vec(), actions)?,
    ))
}
