// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ordered bulk materialization at any height. Keep one pending leaf across
//! routing boundaries so split/coalesce is decided before writing final leaves.

use std::collections::BTreeMap;

use bytes::Bytes;
use lance_core::{Error, Result};

use super::node::{self, FragmentTreeConfig};
use super::store::{LeafUploads, NodeStore, Uploaded};
use crate::format::{Fragment, pb};

#[derive(Default)]
pub(super) struct BulkResult {
    pub children: Vec<pb::FragmentTreeChild>,
    pub io_bytes: u64,
    pub flushes: u64,
    pub splits: u64,
    pub merges: u64,
    pub materialized: u64,
    pub max_flush_depth: u32,
}

impl BulkResult {
    fn with_leaves(self, leaves: Uploaded) -> Self {
        Self {
            children: leaves.children,
            io_bytes: leaves.io_bytes,
            ..self
        }
    }
}

enum PendingLeaf {
    Unchanged(pb::FragmentTreeChild, u64),
    Changed {
        fragments: Vec<Fragment>,
        encoded: Option<Bytes>,
    },
}

impl PendingLeaf {
    async fn bytes(&mut self, store: &NodeStore) -> Result<u64> {
        Ok(match self {
            Self::Unchanged(child, _) => child.object_size,
            Self::Changed { fragments, encoded } => match encoded {
                Some(bytes) => bytes.len() as u64,
                None => {
                    let bytes = store.encode_leaf(fragments).await?;
                    let size = bytes.len() as u64;
                    *encoded = Some(bytes);
                    size
                }
            },
        })
    }

    async fn load(self, store: &NodeStore) -> Result<Vec<Fragment>> {
        match self {
            Self::Unchanged(child, end) => store.read_leaf_in_range(&child, end).await,
            Self::Changed { fragments, .. } => Ok(fragments),
        }
    }

    async fn write(
        self,
        uploads: &mut LeafUploads<'_>,
        config: &FragmentTreeConfig,
        watermark: u64,
        output: &mut BulkResult,
    ) -> Result<()> {
        match self {
            Self::Unchanged(child, _) => uploads.reuse(child),
            Self::Changed { fragments, encoded } => {
                let leaves = uploads
                    .upload_leaves(&fragments, encoded, watermark, config)
                    .await?;
                output.splits += leaves.saturating_sub(1) as u64;
            }
        }
        Ok(())
    }
}

/// Visit old routing in order, carrying each ancestor's actions to its owning
/// leaf. Defer the last output leaf across parent boundaries so a later sparse
/// sibling cannot force a read/rewrite of an already emitted replacement.
/// Unchanged leaves are reused unless coalescing needs them. The caller builds
/// final routing over the returned leaf references, without intermediate nodes.
pub(super) async fn materialize(
    store: &NodeStore,
    config: &FragmentTreeConfig,
    children: Vec<pb::FragmentTreeChild>,
    buffer: Vec<pb::FragmentTreeMutation>,
    watermark: u64,
) -> Result<BulkResult> {
    let mut output = BulkResult::default();
    let mut uploads = LeafUploads::new(store);
    let mut pending: Option<PendingLeaf> = None;
    if children.is_empty() {
        let mut fragments = BTreeMap::new();
        output.materialized = buffer.len() as u64;
        store.apply_verified(&mut fragments, buffer)?;
        if !fragments.is_empty() {
            PendingLeaf::Changed {
                fragments: fragments.into_values().collect(),
                encoded: None,
            }
            .write(&mut uploads, config, watermark, &mut output)
            .await?;
        }
        return Ok(output.with_leaves(uploads.finish().await?));
    }
    let mut sequences = node::DecodedSequences::new(&buffer);
    let buckets = node::partition_buffer_by_child(&children, buffer);
    node::validate_routed(&children, &buckets, node::ROOT_EXCLUSIVE_END)?;
    let mut stack: Vec<_> = node::with_exclusive_ends(children, node::ROOT_EXCLUSIVE_END)
        .zip(buckets)
        .map(|((child, end), actions)| (child, actions, end, 0))
        .rev()
        .collect();
    while let Some((child, actions, end, depth)) = stack.pop() {
        if child.height > 0 {
            let mut internal = store.read_internal_in_range(&child, end).await?;
            if internal.children.is_empty() {
                return Err(Error::invalid_input(format!(
                    "fragment metadata bulk encountered childless interior {} at height {}",
                    child.path, child.height
                )));
            }
            sequences.admit(&child.path, &internal.buffer)?;
            internal.buffer.extend(actions);
            let buckets = node::partition_buffer_by_child(&internal.children, internal.buffer);
            node::validate_routed(&internal.children, &buckets, end)?;
            stack.extend(
                node::with_exclusive_ends(internal.children, end)
                    .zip(buckets)
                    .map(|((child, end), actions)| (child, actions, end, depth + 1))
                    .rev(),
            );
            continue;
        }
        let mut current = if actions.is_empty() {
            PendingLeaf::Unchanged(child, end)
        } else {
            output.flushes += 1;
            output.max_flush_depth = output.max_flush_depth.max(depth);
            output.materialized += actions.len() as u64;
            let mut fragments: BTreeMap<_, _> = store
                .read_leaf_in_range(&child, end)
                .await?
                .into_iter()
                .map(|f| (f.id, f))
                .collect();
            store.apply_verified(&mut fragments, actions)?;
            if fragments.is_empty() {
                continue;
            }
            PendingLeaf::Changed {
                fragments: fragments.into_values().collect(),
                encoded: None,
            }
        };
        if let Some(mut previous) = pending.take() {
            let previous_bytes = previous.bytes(store).await?;
            let current_bytes = current.bytes(store).await?;
            let small = previous_bytes <= config.leaf_merge_floor()
                || current_bytes <= config.leaf_merge_floor();
            let fits = previous_bytes
                .checked_add(current_bytes)
                .is_some_and(|bytes| bytes <= config.leaf_coalesce_ceiling());
            if small && fits {
                let mut fragments = previous.load(store).await?;
                fragments.extend(current.load(store).await?);
                pending = Some(PendingLeaf::Changed {
                    fragments,
                    encoded: None,
                });
                output.merges += 1;
                continue;
            }
            previous
                .write(&mut uploads, config, watermark, &mut output)
                .await?;
        }
        pending = Some(current);
    }
    if let Some(leaf) = pending {
        leaf.write(&mut uploads, config, watermark, &mut output)
            .await?;
    }
    Ok(output.with_leaves(uploads.finish().await?))
}
