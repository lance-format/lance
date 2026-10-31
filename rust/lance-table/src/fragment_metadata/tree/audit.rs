// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Audit one snapshot against the objects it references in storage.

use super::publication::{CheckedRoot, checked_root};
use super::*;
use crate::fragment_metadata::validation;

impl FragmentTree {
    /// Check this snapshot against every object it references in storage, to
    /// catch corruption that ordinary reads let through.
    ///
    /// An inline root comes from this snapshot's descriptor. An external root
    /// and every interior and leaf come from storage, not from a cache or the
    /// resident inventory. It checks that
    ///
    /// - the descriptor, root and suffix pass every check opening runs,
    /// - every interior and leaf matches its parent reference and its
    ///   inherited range, foreign subtrees included,
    /// - no node object is reached from two positions. The fence and range
    ///   checks reject sibling and cousin aliases, and the height check
    ///   rejects a cycle,
    /// - every interior sequence and leaf watermark below the root lies below
    ///   the root's frontier,
    /// - every buffered action, unsquashed, routes inside its child's range and
    ///   above its leaf's watermark,
    /// - no action sequence repeats across the buffers of the tree,
    /// - every stored id and buffered target lies below the
    ///   `next_fragment_id` this snapshot was opened with,
    /// - replaying every buffered action reproduces its stored deltas,
    /// - every data file of the replayed records passes
    ///   [`DataFile::validate`](crate::format::DataFile::validate),
    /// - the replayed fragment, physical-row and visible-row totals equal
    ///   [`Self::count_fragments`], [`Self::count_rows`] and
    ///   [`Self::count_visible_rows`],
    /// - and a [`Self::scan`] of this snapshot yields the replayed records.
    ///   The scan reads through the caches and the resident inventory, so it
    ///   checks that they serve what storage holds.
    ///
    /// It does not check writer policy such as fanout, byte budgets, inline or
    /// external roots and drain timing, because a tree written under other
    /// settings is valid. It also skips unreachable objects, other versions,
    /// the contents of data and deletion files and Version Manifest fields
    /// outside the descriptor. It cannot check that a leaf holds every mutation
    /// up to its watermark without older versions.
    ///
    /// It holds every persisted record in memory and reads each object once.
    /// The scan reads every interior and leaf again only when the snapshot is
    /// lazy and its caches are cold. Reads and commits never call it.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # async fn example(tree: &FragmentTree) -> lance_core::Result<()> {
    /// tree.validate().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn validate(&self) -> Result<()> {
        let records = self.persisted_tree().await?.replayed_records().await?;
        self.check_served(&records).await
    }

    /// Open this snapshot's descriptor like [`Self::open_snapshot`], without
    /// caches or squashing. Squashing could hide a stale action or deltas that
    /// cancel within a run.
    async fn persisted_tree(&self) -> Result<Self> {
        let descriptor = &self.descriptor;
        // A root read consults no cache, and the uncached store needs the
        // root's frontier.
        let CheckedRoot {
            root,
            total_fragments,
            total_rows,
            visible_rows,
        } = checked_root(&self.store, descriptor, self.version, self.next_fragment_id).await?;
        let mut buffer = root.buffer;
        buffer.extend(descriptor.mutations_since_root.iter().cloned());
        Ok(Self {
            // Interiors and leaves below the root were written before the
            // root, so the format bounds them by its frontier.
            store: self.store.uncached(root.next_action_sequence),
            config: self.config.clone(),
            version: self.version,
            children: root.children,
            buffer,
            buffer_index: OnceLock::new(),
            next_action_sequence: descriptor.next_action_sequence,
            contiguous_from: root.next_action_sequence,
            total_fragments,
            total_rows,
            next_fragment_id: self.next_fragment_id,
            visible_rows,
            descriptor: descriptor.clone(),
            resident: None,
        })
    }

    async fn replayed_records(&self) -> Result<Vec<Fragment>> {
        let (mut records, actions) = self.stored_and_routed().await?;
        self.store.apply_verified(&mut records, actions)?;
        let records: Vec<Fragment> = records.into_values().collect();
        // Replay converts buffered files without validating them.
        for file in records.iter().flat_map(Fragment::referenced_lance_files) {
            file.validate(self.store.base())?;
        }
        Ok(records)
    }

    /// Require this snapshot's counts and scan to match `persisted`.
    async fn check_served(&self, persisted: &[Fragment]) -> Result<()> {
        let (rows, visible_rows) = content::fragment_row_counts(persisted)?;
        let stored = (persisted.len() as u64, rows, visible_rows);
        let counted = (self.total_fragments, self.total_rows, self.visible_rows);
        if counted != stored {
            return Err(validation::corrupt(format!(
                "Fragment metadata tree version {} counts {} fragments, {} physical rows and {} \
                 visible rows, but its persisted records hold {} fragments, {} physical rows \
                 and {} visible rows",
                self.version, counted.0, counted.1, counted.2, stored.0, stored.1, stored.2
            )));
        }
        let served = Arc::new(self.clone()).scan(0);
        if let Some(id) = first_difference(served, persisted).await? {
            return Err(validation::corrupt(format!(
                "Fragment metadata tree version {} serves fragment {id} differently from its \
                 persisted records",
                self.version
            )));
        }
        Ok(())
    }
}

/// The smallest id at which `served` and `persisted` differ. Both must be in
/// ascending id order.
async fn first_difference(
    mut served: BoxStream<'_, Result<Fragment>>,
    persisted: &[Fragment],
) -> Result<Option<u64>> {
    let mut persisted = persisted.iter();
    loop {
        match (served.try_next().await?, persisted.next()) {
            (None, None) => return Ok(None),
            (Some(left), Some(right)) if left == *right => {}
            (Some(left), Some(right)) => return Ok(Some(left.id.min(right.id))),
            (Some(extra), None) => return Ok(Some(extra.id)),
            (None, Some(missing)) => return Ok(Some(missing.id)),
        }
    }
}
